"""
============================================================================
INFERENCE.PY — Run Predictions with Your Fine-Tuned Gemma Model
============================================================================

PURPOSE:
    After training, use this script to generate product recommendations
    from new review text. It loads the fine-tuned LoRA adapters and
    generates responses.

USAGE:
    # Single prediction
    python inference.py --model_dir ./outputs/run_XXXXXXXX/final_model \
                        --prompt "This phone has amazing battery life. Rating: 5"
    
    # Interactive mode
    python inference.py --model_dir ./outputs/run_XXXXXXXX/final_model \
                        --interactive
    
    # From Python
    from inference import load_inference_model, generate_recommendation
    model, tokenizer = load_inference_model("./outputs/run_XXXXXXXX/final_model")
    result = generate_recommendation(model, tokenizer, "Great product! Rating: 5")

HOW INFERENCE WORKS:
    1. Load the base Gemma model (quantized to 4-bit)
    2. Load the LoRA adapter weights from training
    3. Merge adapters into the base model (optional, for speed)
    4. Tokenize the input prompt
    5. Generate tokens autoregressively (one at a time)
    6. Decode generated tokens back to text

AUTOREGRESSIVE GENERATION EXPLAINED:
    LLMs generate text one token at a time:
    
    Step 1: Input "The product is"
    Step 2: Model predicts next token → "great"
    Step 3: Input "The product is great"
    Step 4: Model predicts next token → "for"
    Step 5: Input "The product is great for"
    ... and so on until max_new_tokens or end-of-sequence token.
"""

import os
import sys
import argparse
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from peft import PeftModel

from config import TrainingConfig
from data_loader import INFERENCE_TEMPLATE


def load_inference_model(
    model_dir: str,
    merge_adapters: bool = False,
    config: Optional[TrainingConfig] = None,
):
    """
    Load the fine-tuned model for inference.
    
    TWO APPROACHES:
    
    1. PEFT Model (merge_adapters=False):
       - Loads base model + adapter separately
       - Adapter applied dynamically during forward pass
       - Uses slightly more memory at runtime
       - Can easily switch between different adapters
       - RECOMMENDED for experimentation
       
    2. Merged Model (merge_adapters=True):
       - Merges adapter weights INTO the base model
       - Creates a single unified model
       - Slightly faster inference (no adapter overhead)
       - Can't switch adapters anymore
       - Good for deployment/production
    
    Args:
        model_dir: Path to the saved fine-tuned model (adapter weights)
        merge_adapters: Whether to merge LoRA weights into the base model
        config: TrainingConfig (uses defaults if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if config is None:
        config = TrainingConfig()

    print("\n" + "=" * 60)
    print("  LOADING FINE-TUNED MODEL FOR INFERENCE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load quantization config (same as training)
    # ------------------------------------------------------------------
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=config.compute_dtype,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )

    # ------------------------------------------------------------------
    # Step 2: Load the base model
    # ------------------------------------------------------------------
    print(f"\n📦 Loading base model: {config.model_name}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    )

    # ------------------------------------------------------------------
    # Step 3: Load LoRA adapter weights
    # ------------------------------------------------------------------
    print(f"\n🔌 Loading LoRA adapter from: {model_dir}")

    model = PeftModel.from_pretrained(
        base_model,
        model_dir,
        is_trainable=False,  # Inference only, no gradient computation
    )

    # ------------------------------------------------------------------
    # Step 4: Optionally merge adapters
    # ------------------------------------------------------------------
    if merge_adapters:
        print("\n🔗 Merging LoRA adapters into base model...")
        # merge_and_unload() does:
        # 1. Computes W_new = W_original + (alpha/r) × A × B
        # 2. Replaces the original linear layers with the merged weights
        # 3. Removes the PEFT wrapper (returns a vanilla model)
        model = model.merge_and_unload()
        print("   ✅ Adapters merged. Model is now a standard Gemma model.")
    else:
        print("   Using PEFT model (adapters applied dynamically)")

    # Set model to evaluation mode
    # This disables dropout layers (which you don't want during inference)
    model.eval()

    # ------------------------------------------------------------------
    # Step 5: Load tokenizer
    # ------------------------------------------------------------------
    print(f"\n📝 Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,  # Load from fine-tuned dir (may have special tokens)
        trust_remote_code=config.trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For generation, padding should be on the LEFT side
    # WHY LEFT: During generation, the model generates tokens at the END.
    # If padding is on the right, the model would generate after padding
    # tokens, which produces garbage. Left padding puts real tokens at
    # the end where generation happens.
    tokenizer.padding_side = "left"

    print(f"   ✅ Model ready for inference!")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"   GPU memory: {mem:.2f} GB")

    return model, tokenizer


def generate_recommendation(
    model,
    tokenizer,
    review: str,
    rating: float = 3.0,
    title: str = "Product",
    category: str = "General",
    config: Optional[TrainingConfig] = None,
    stream: bool = False,
) -> str:
    """
    Generate a product recommendation from a review.
    
    GENERATION PARAMETERS EXPLAINED:
    
    max_new_tokens:
        Maximum number of tokens to generate. 256 tokens ≈ 200 words.
        
    temperature:
        Controls randomness. Lower = more deterministic, higher = more creative.
        
        HOW IT WORKS:
          Model outputs a probability for each possible next token.
          Temperature divides the logits before softmax:
            probabilities = softmax(logits / temperature)
          
          temperature=0.1: Probabilities become very peaked (almost always 
                            picks the highest-prob token). Very repetitive.
          temperature=0.7: Moderate randomness. Good balance.
          temperature=1.0: Standard softmax. More diverse outputs.
          temperature=2.0: Very flat distribution. Mostly random.
          
    top_p (nucleus sampling):
        Instead of considering ALL possible tokens, only consider the smallest
        set of tokens whose cumulative probability exceeds top_p.
        
        Example with top_p=0.9:
          Token A: 50% prob  ← included (cumsum: 50%)
          Token B: 30% prob  ← included (cumsum: 80%)
          Token C: 15% prob  ← included (cumsum: 95% > 90%)
          Token D: 5% prob   ← EXCLUDED
          
        This filters out very unlikely tokens that could produce nonsense.
        
    top_k:
        Only consider the top K most likely tokens at each step.
        top_k=50 means: choose from the 50 most likely tokens.
        top_k=1 means: always pick the single most likely token (greedy).
        
    repetition_penalty:
        Penalizes tokens that have already appeared in the output.
        1.0 = no penalty. 1.2 = 20% less likely to repeat a token.
        Prevents the model from getting stuck in loops like
        "great great great great great..."
        
    do_sample:
        True = sample tokens from the probability distribution
        False = always pick the most likely token (greedy decoding)
        Must be True for temperature, top_p, top_k to have any effect.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        review: Product review text
        rating: Product rating (1-5)
        title: Product title
        category: Product category
        config: TrainingConfig (uses defaults if None)
        stream: Whether to stream output token by token
        
    Returns:
        Generated recommendation text
    """
    if config is None:
        config = TrainingConfig()

    # Format the prompt using our inference template
    # (same format as training, but without the model response)
    prompt = INFERENCE_TEMPLATE.format(
        category=category,
        title=title,
        rating=rating,
        review=review,
    )

    # Tokenize the input
    # return_tensors="pt" = return PyTorch tensors
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_seq_length,
    )

    # Move input tensors to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set up streaming (optional — prints tokens as they're generated)
    streamer = None
    if stream:
        # TextStreamer prints each token as it's generated (like ChatGPT)
        streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    # GENERATE!
    # torch.no_grad() disables gradient computation (not needed for inference)
    # This saves memory and speeds up generation.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,

            # How many new tokens to generate (output length)
            max_new_tokens=config.max_new_tokens,

            # Sampling parameters
            do_sample=True,           # Enable sampling (vs. greedy)
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,

            # Repetition control
            repetition_penalty=1.15,  # Slightly penalize repeating tokens

            # Stop at end-of-turn token
            # This tells the model to stop generating when it produces
            # the <end_of_turn> token (end of the model's response)
            eos_token_id=tokenizer.eos_token_id,

            # Streaming
            streamer=streamer,
        )

    # Decode the generated tokens to text
    # We need to strip the input prompt from the output because
    # model.generate() returns input + generated tokens
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def interactive_mode(model, tokenizer, config: TrainingConfig):
    """
    Interactive CLI mode for testing the model.
    
    Type a product review, and the model generates a recommendation.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "=" * 60)
    print("  🤖 INTERACTIVE MODE — Ask for product recommendations!")
    print("=" * 60)
    print("  Type a product review and rating to get a recommendation.")
    print("  Format: <review text> | <rating 1-5>")
    print("  Example: Great battery life and camera | 5")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("📝 Your review: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        if not user_input:
            continue

        # Parse rating from input (format: "review text | rating")
        if "|" in user_input:
            parts = user_input.rsplit("|", 1)
            review = parts[0].strip()
            try:
                rating = float(parts[1].strip())
            except ValueError:
                rating = 3.0
        else:
            review = user_input
            rating = 3.0

        print(f"\n🤔 Generating recommendation (rating: {rating}/5)...")
        print("-" * 60)

        response = generate_recommendation(
            model=model,
            tokenizer=tokenizer,
            review=review,
            rating=rating,
            config=config,
            stream=True,  # Stream output for interactive feel
        )

        print("-" * 60)
        print()


def main():
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Gemma model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python inference.py --model_dir ./outputs/run_xxx/final_model \\
    --prompt "This laptop has great performance but poor battery life" \\
    --rating 3

  # Interactive mode
  python inference.py --model_dir ./outputs/run_xxx/final_model --interactive

  # With custom generation settings
  python inference.py --model_dir ./outputs/run_xxx/final_model \\
    --prompt "Best phone ever!" --rating 5 \\
    --temperature 0.5 --max_tokens 200
        """,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the saved fine-tuned model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Product review text for single prediction",
    )
    parser.add_argument(
        "--rating",
        type=float,
        default=3.0,
        help="Product rating (1-5). Default: 3.0",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Product",
        help="Product title. Default: 'Product'",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="General",
        help="Product category. Default: 'General'",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode for continuous testing",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA adapters into base model (faster inference)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (overrides config)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens to generate (overrides config)",
    )

    args = parser.parse_args()

    # Validate
    if not args.prompt and not args.interactive:
        parser.error("Either --prompt or --interactive must be specified")

    if not os.path.exists(args.model_dir):
        parser.error(f"Model directory not found: {args.model_dir}")

    # Load config and apply overrides
    config = TrainingConfig()
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.max_tokens is not None:
        config.max_new_tokens = args.max_tokens

    # Load model
    model, tokenizer = load_inference_model(
        model_dir=args.model_dir,
        merge_adapters=args.merge,
        config=config,
    )

    if args.interactive:
        interactive_mode(model, tokenizer, config)
    else:
        print(f"\n📝 Review: {args.prompt}")
        print(f"   Rating: {args.rating}/5")
        print(f"\n🤖 Generating recommendation...\n")
        print("-" * 60)

        response = generate_recommendation(
            model=model,
            tokenizer=tokenizer,
            review=args.prompt,
            rating=args.rating,
            title=args.title,
            category=args.category,
            config=config,
            stream=True,
        )

        print("-" * 60)
        print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
