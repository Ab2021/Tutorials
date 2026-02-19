"""
============================================================================
EVALUATE.PY — Evaluation Metrics for Fine-Tuned Gemma Model
============================================================================

PURPOSE:
    Evaluates the fine-tuned model's output quality using standard NLG
    (Natural Language Generation) metrics:
    • ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    • BLEU (Bilingual Evaluation Understudy)

USAGE:
    python evaluate.py --model_dir ./outputs/run_XXXXXXXX/final_model

    Results are printed to console and saved to a JSON file.

METRICS EXPLAINED:

    ROUGE-1 (Unigram overlap):
        Measures word-level overlap between generated and reference text.
        Example:
          Reference: "buy this great product"
          Generated: "buy this amazing product"
          ROUGE-1 = 3 matching words / 4 total = 0.75
        
    ROUGE-2 (Bigram overlap):
        Measures 2-word phrase overlap.
        Example:
          Reference: "buy this great product"
          Generated: "buy this amazing product"  
          Bigrams ref: {"buy this", "this great", "great product"}
          Bigrams gen: {"buy this", "this amazing", "amazing product"}
          ROUGE-2 = 1 match / 3 = 0.33
        
    ROUGE-L (Longest Common Subsequence):
        Finds the longest sequence of words that appear in both texts
        in the same order (not necessarily consecutively).
        More flexible than ROUGE-1/2 — captures sentence structure.
        
    BLEU (Bilingual Evaluation Understudy):
        Originally designed for machine translation, also used for text gen.
        Measures n-gram precision (how much of the generated text appears
        in the reference) with a brevity penalty.
        
    WHAT ARE GOOD SCORES?
        There's no universal threshold — it depends on the task:
        • Summarization: ROUGE-L > 0.30 is decent
        • Translation: BLEU > 0.30 is decent
        • Creative generation: Lower scores are expected (many valid outputs)
        
        For our recommendation task, scores > 0.20 indicate the model has
        learned the desired output format and content patterns.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from config import TrainingConfig
from inference import load_inference_model, generate_recommendation
from data_loader import load_and_prepare_data, clean_text


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict:
    """
    Compute ROUGE scores between predictions and references.
    
    ROUGE INTERNALS:
    
    ROUGE works by comparing n-gram overlap between the generated text
    (prediction) and the expected text (reference).
    
    For each pair, it computes:
    - Precision: What fraction of generated n-grams appear in the reference?
    - Recall: What fraction of reference n-grams appear in the generated text?
    - F1: Harmonic mean of precision and recall
    
    We report F1 scores because they balance precision and recall:
    - High precision, low recall = generated text is accurate but incomplete
    - Low precision, high recall = generated text captures everything but adds noise
    - High F1 = good balance of accuracy and completeness
    
    Args:
        predictions: List of generated texts
        references: List of reference (expected) texts
        
    Returns:
        Dictionary of ROUGE scores
    """
    from rouge_score import rouge_scorer

    # Create scorer that computes ROUGE-1, ROUGE-2, and ROUGE-L
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,  # Stem words before comparing ("running" → "run")
    )

    # Accumulate scores across all examples
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue

        scores = scorer.score(ref, pred)

        # .fmeasure = F1 score (harmonic mean of precision and recall)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    # Average across all examples
    results = {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
        "num_examples": len(rouge1_scores),
    }

    return results


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute corpus-level BLEU score.
    
    BLEU INTERNALS:
    
    1. Tokenize both texts into words
    2. Count matching n-grams (n=1,2,3,4) between prediction and reference
    3. Compute precision for each n-gram size:
       precision_n = matched_n_grams / total_n_grams_in_prediction
    4. Compute geometric mean of precisions (equally weights all n-gram sizes)
    5. Apply brevity penalty to discourage overly short outputs:
       BP = exp(1 - ref_length / pred_length) if pred shorter than ref, else 1.0
    6. BLEU = BP × geometric_mean(precisions)
    
    CORPUS BLEU vs SENTENCE BLEU:
    - Sentence BLEU: computed for each example individually, then averaged
      Problem: a single short prediction can make individual BLEU = 0
    - Corpus BLEU: aggregates n-gram counts across ALL examples, then computes
      score once. More stable and the standard approach.
    
    We use CORPUS BLEU for more reliable scores.
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        
    Returns:
        BLEU score (float between 0 and 1)
    """
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    # Download punkt tokenizer if not already present
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("   Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab", quiet=True)

    # Tokenize texts into word lists
    # BLEU expects: references as list of list of lists (each ref can have multiple valid translations)
    # predictions as list of lists (each prediction tokenized)
    tokenized_refs = []
    tokenized_preds = []

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue

        # Tokenize into words
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())

        # BLEU format: references is [[ref_tokens]], predictions is [pred_tokens]
        # Each reference position can have multiple valid translations, hence the nesting
        tokenized_refs.append([ref_tokens])  # One reference per example
        tokenized_preds.append(pred_tokens)

    if not tokenized_refs:
        return 0.0

    # SmoothingFunction handles the case where an n-gram has zero matches
    # Without smoothing, a single missing 4-gram makes the entire BLEU = 0
    # Method 4 (exponential smoothing) is the most commonly used
    smoothing = SmoothingFunction().method4

    try:
        bleu = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            smoothing_function=smoothing,
        )
    except Exception as e:
        print(f"   ⚠️  BLEU computation error: {e}")
        bleu = 0.0

    return bleu


def extract_model_response(formatted_text: str) -> str:
    """
    Extract the model's response from a formatted prompt.
    
    Our training data format is:
    <start_of_turn>user ... <end_of_turn>
    <start_of_turn>model ... <end_of_turn>
    
    We extract the text between <start_of_turn>model and <end_of_turn>.
    """
    marker = "<start_of_turn>model\n"
    end_marker = "<end_of_turn>"

    start_idx = formatted_text.find(marker)
    if start_idx == -1:
        return formatted_text

    start_idx += len(marker)
    end_idx = formatted_text.find(end_marker, start_idx)

    if end_idx == -1:
        return formatted_text[start_idx:]

    return formatted_text[start_idx:end_idx].strip()


def extract_review_info(formatted_text: str) -> dict:
    """Extract review, rating, title, category from formatted prompt."""
    info = {
        "review": "",
        "rating": 3.0,
        "title": "Product",
        "category": "General",
    }

    # Extract review text
    review_start = formatted_text.find("Review: ")
    if review_start != -1:
        review_end = formatted_text.find("\n<end_of_turn>", review_start)
        if review_end != -1:
            info["review"] = formatted_text[review_start + 8:review_end].strip()

    # Extract rating
    rating_start = formatted_text.find("Rating: ")
    if rating_start != -1:
        rating_end = formatted_text.find("/5", rating_start)
        if rating_end != -1:
            try:
                info["rating"] = float(formatted_text[rating_start + 8:rating_end])
            except ValueError:
                pass

    # Extract title
    title_start = formatted_text.find("Product Title: ")
    if title_start != -1:
        title_end = formatted_text.find("\n", title_start)
        if title_end != -1:
            info["title"] = formatted_text[title_start + 15:title_end].strip()

    # Extract category
    cat_start = formatted_text.find("Product Category: ")
    if cat_start != -1:
        cat_end = formatted_text.find("\n", cat_start)
        if cat_end != -1:
            info["category"] = formatted_text[cat_start + 18:cat_end].strip()

    return info


def evaluate(
    model_dir: str,
    max_samples: int = 50,
    config: Optional[TrainingConfig] = None,
) -> Dict:
    """
    Run full evaluation of the fine-tuned model.
    
    EVALUATION FLOW:
    1. Load fine-tuned model
    2. Load test dataset
    3. For each test example:
       a. Extract the review + rating from the formatted prompt
       b. Generate a recommendation using the model
       c. Extract the reference response from the formatted prompt
    4. Compute ROUGE and BLEU scores
    5. Report results and save to JSON
    
    Args:
        model_dir: Path to the saved fine-tuned model
        max_samples: Maximum number of test examples to evaluate
        config: TrainingConfig instance
        
    Returns:
        Dictionary of evaluation results
    """
    if config is None:
        config = TrainingConfig()

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    print("\n🤖 Loading fine-tuned model...")
    model, tokenizer = load_inference_model(model_dir, config=config)

    # ------------------------------------------------------------------
    # Step 2: Load test data
    # ------------------------------------------------------------------
    print("\n📊 Loading test dataset...")
    _, _, _, test_dataset = load_and_prepare_data(config)

    # Limit samples
    if max_samples and len(test_dataset) > max_samples:
        test_dataset = test_dataset.select(range(max_samples))
    print(f"   Evaluating on {len(test_dataset)} examples...")

    # ------------------------------------------------------------------
    # Step 3: Generate predictions
    # ------------------------------------------------------------------
    print("\n🔮 Generating predictions...")
    predictions = []
    references = []

    for i, example in enumerate(tqdm(test_dataset, desc="Generating")):
        formatted_text = example["text"]

        # Extract reference response (what we trained the model to output)
        reference = extract_model_response(formatted_text)
        references.append(reference)

        # Extract input information
        info = extract_review_info(formatted_text)

        # Generate model's prediction
        try:
            prediction = generate_recommendation(
                model=model,
                tokenizer=tokenizer,
                review=info["review"],
                rating=info["rating"],
                title=info["title"],
                category=info["category"],
                config=config,
                stream=False,
            )
            predictions.append(prediction)
        except Exception as e:
            print(f"\n   ⚠️  Error on example {i}: {e}")
            predictions.append("")

        # Print first few examples for manual inspection
        if i < 3:
            print(f"\n   --- Example {i + 1} ---")
            print(f"   Review: {info['review'][:100]}...")
            print(f"   Rating: {info['rating']}")
            print(f"   Reference (first 150 chars): {reference[:150]}...")
            print(f"   Predicted (first 150 chars): {prediction[:150]}...")

    # ------------------------------------------------------------------
    # Step 4: Compute metrics
    # ------------------------------------------------------------------
    print("\n📊 Computing metrics...")

    rouge_results = compute_rouge_scores(predictions, references)
    bleu_score = compute_bleu_score(predictions, references)

    results = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleu": bleu_score,
        "num_examples": len(predictions),
        "model_dir": model_dir,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": config.model_name,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        },
    }

    # ------------------------------------------------------------------
    # Step 5: Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n   Model:   {model_dir}")
    print(f"   Samples: {results['num_examples']}")
    print()
    print(f"   ┌─────────────┬──────────┐")
    print(f"   │ Metric      │ Score    │")
    print(f"   ├─────────────┼──────────┤")
    print(f"   │ ROUGE-1     │ {results['rouge1']:.4f}   │")
    print(f"   │ ROUGE-2     │ {results['rouge2']:.4f}   │")
    print(f"   │ ROUGE-L     │ {results['rougeL']:.4f}   │")
    print(f"   │ BLEU        │ {results['bleu']:.4f}   │")
    print(f"   └─────────────┴──────────┘")

    print(f"\n   Score Interpretation:")
    for metric, score in [("ROUGE-L", results["rougeL"]), ("BLEU", results["bleu"])]:
        if score > 0.5:
            quality = "🟢 Excellent"
        elif score > 0.3:
            quality = "🟡 Good"
        elif score > 0.15:
            quality = "🟠 Moderate"
        else:
            quality = "🔴 Needs improvement"
        print(f"   {metric}: {quality}")

    # ------------------------------------------------------------------
    # Step 6: Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(model_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   💾 Results saved to: {results_path}")

    print("\n" + "=" * 60)

    return results


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Gemma model with ROUGE and BLEU metrics",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the saved fine-tuned model directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum number of test examples to evaluate. Default: 50",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        parser.error(f"Model directory not found: {args.model_dir}")

    evaluate(
        model_dir=args.model_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
