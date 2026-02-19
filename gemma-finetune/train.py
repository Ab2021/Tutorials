"""
============================================================================
TRAIN.PY — Main Training Script for Gemma Fine-Tuning
============================================================================

PURPOSE:
    This is the main entry point for fine-tuning Gemma. It orchestrates:
    1. Loading configuration
    2. Setting up the model with QLoRA
    3. Loading and preparing the dataset
    4. Running supervised fine-tuning with SFTTrainer
    5. Saving the fine-tuned model

USAGE:
    python train.py

    Or with custom config overrides:
    python train.py  (edit config.py before running)

WHAT IS SFTTrainer?
    SFTTrainer (Supervised Fine-Tuning Trainer) from the TRL library is a
    specialized trainer designed for fine-tuning large language models.
    
    Built on top of HuggingFace's Trainer, it adds:
    • Chat/instruction template handling
    • Sequence packing (fitting multiple short examples into one sequence)
    • Native PEFT/LoRA integration
    • Efficient tokenization of the "text" column

TRAINING LOOP EXPLAINED:
    For each batch of examples:
    1. FORWARD PASS:
       - Input tokens go through the model
       - Model predicts next token at each position
       - Loss = cross-entropy between predicted and actual next tokens
       
    2. BACKWARD PASS:
       - Compute gradients of the loss w.r.t. LoRA parameters
       - (Base model parameters are frozen — no gradients computed for them)
       
    3. OPTIMIZER STEP:
       - Update LoRA parameters using AdamW optimizer
       - Apply learning rate schedule
       - Apply gradient clipping
       
    4. LOGGING:
       - Report loss, learning rate, GPU memory usage
       
    5. CHECKPOINTING:
       - Save model weights periodically
"""

import os
import sys
import warnings
from datetime import datetime

import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from config import TrainingConfig
from model_setup import load_model_and_tokenizer
from data_loader import load_and_prepare_data


def get_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """
    Create the TrainingArguments that control every aspect of training.
    
    EVERY PARAMETER EXPLAINED:
    
    output_dir:
        Where to save checkpoints, logs, and the final model.
        Created automatically if it doesn't exist.
        
    num_train_epochs:
        Number of complete passes through the training data.
        Each epoch sees every training example exactly once.
        
    per_device_train_batch_size:
        How many examples to process on each GPU per forward pass.
        If you have 2 GPUs with batch_size=4, total batch = 8 per step.
        REDUCE this if you get CUDA OOM errors.
        
    per_device_eval_batch_size:
        Batch size for evaluation. Can be larger than training because
        we don't store gradients during eval (less VRAM needed).
        
    gradient_accumulation_steps:
        Accumulate gradients over N batches before updating weights.
        Simulates larger batch size: effective_batch = batch × accumulation.
        
    learning_rate:
        Initial learning rate for the optimizer.
        This is THE most important hyperparameter. Too high → divergence.
        Too low → slow convergence.
        
    weight_decay:
        L2 regularization. Adds penalty for large weights.
        Helps prevent overfitting.
        
    warmup_ratio:
        Fraction of steps for linear warmup. Learning rate starts at 0
        and linearly increases to `learning_rate` over this many steps.
        
    lr_scheduler_type:
        How LR changes over training. "cosine" starts high, decreases
        following a cosine curve. Empirically better than linear.
        
    max_grad_norm:
        Gradient clipping. If gradient norm > this value, scale it down.
        Prevents exploding gradients from bad batches.
        
    fp16 / bf16:
        Mixed precision training. Does most computations in 16-bit float
        instead of 32-bit, cutting memory usage and increasing speed.
        bf16 is more numerically stable but requires Ampere+ GPU.
        
    logging_steps:
        Print loss and other metrics every N training steps.
        
    save_steps:
        Save a checkpoint every N steps. Checkpoints allow resuming
        training if it crashes.
        
    eval_steps / evaluation_strategy:
        Run validation every N steps. "steps" means based on step count.
        "epoch" would run validation after each epoch instead.
        
    save_total_limit:
        Keep at most N checkpoints. Older ones are deleted to save disk.
        
    seed:
        Random seed for reproducibility.
        
    gradient_checkpointing:
        Trades compute for memory. Instead of storing all activations
        in memory for the backward pass, recompute them on the fly.
        Saves ~30-50% VRAM at ~20% speed cost.
        
    optim:
        Optimizer to use. "paged_adamw_8bit" is an 8-bit version of
        AdamW that uses paging (spills to CPU if GPU runs out of memory).
        This saves ~2 GB of VRAM compared to standard AdamW.
        
        Other options:
        - "adamw_torch" — Standard PyTorch AdamW (uses most memory)
        - "adamw_8bit"  — 8-bit AdamW (saves ~2 GB, no paging)
        - "paged_adamw_8bit" — 8-bit with paging (best for low VRAM)
        - "adafactor" — Memory-efficient, no momentum. Less stable.
        
    report_to:
        Where to send training metrics. "none" = console only.
        "wandb" = Weights & Biases. "tensorboard" = TensorBoard.
        
    dataloader_pin_memory:
        Pin data tensors in page-locked CPU memory for faster GPU transfer.
        Always True unless you have very little CPU RAM.
        
    remove_unused_columns:
        Remove dataset columns not used by the model.
        Must be False when using SFTTrainer with the "text" column format.
    """
    # Create output directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"run_{timestamp}")

    training_args = TrainingArguments(
        # Output
        output_dir=run_dir,

        # Training schedule
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,

        # Optimizer — 8-bit paged AdamW for memory efficiency
        optim="paged_adamw_8bit",

        # Mixed precision
        fp16=config.fp16,
        bf16=config.bf16,

        # Logging
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=config.logging_steps,
        logging_first_step=True,  # Log the very first step

        # Checkpointing
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy="steps",

        # Evaluation
        eval_steps=config.eval_steps,
        eval_strategy="steps",

        # Gradient checkpointing
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Misc
        seed=config.seed,
        report_to=config.report_to,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Required for SFTTrainer

        # Load best model at end based on eval loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    return training_args


def train():
    """
    Main training function. Orchestrates the entire fine-tuning pipeline.
    
    EXECUTION FLOW:
    
    ┌─────────────────┐
    │ 1. Load Config   │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 2. Load Model    │  ← QLoRA quantization + LoRA adapters
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 3. Load Data     │  ← Download → Clean → Format → Tokenize
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 4. Create Trainer│  ← SFTTrainer with all training args
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 5. TRAIN!        │  ← The main training loop
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 6. Save Model    │  ← Save LoRA adapters (not the full model)
    └─────────────────┘
    """
    print("\n" + "🚀" * 30)
    print("  GEMMA FINE-TUNING — TRAINING PIPELINE")
    print("🚀" * 30)

    # ==================================================================
    # Step 1: Load configuration
    # ==================================================================
    print("\n📋 Step 1: Loading configuration...")
    config = TrainingConfig()
    print(config)  # Print all hyperparameters

    # ==================================================================
    # Step 2: Load model with QLoRA
    # ==================================================================
    print("\n🤖 Step 2: Loading model with QLoRA...")
    model, tokenizer = load_model_and_tokenizer(config)

    # ==================================================================
    # Step 3: Load and prepare data
    # ==================================================================
    print("\n📊 Step 3: Loading and preparing dataset...")
    _, train_dataset, val_dataset, test_dataset = load_and_prepare_data(config)

    # ==================================================================
    # Step 4: Create training arguments
    # ==================================================================
    print("\n⚙️  Step 4: Setting up training arguments...")
    training_args = get_training_arguments(config)
    print(f"   Output directory: {training_args.output_dir}")
    print(f"   Effective batch size: {config.effective_batch_size}")
    print(f"   Total training steps: ~{len(train_dataset) * config.num_epochs // config.effective_batch_size}")

    # ==================================================================
    # Step 5: Create SFTTrainer
    # ==================================================================
    print("\n🏋️  Step 5: Creating SFTTrainer...")

    # SFTTrainer EXPLAINED:
    #
    # SFTTrainer is a subclass of HuggingFace Trainer that adds:
    #
    # 1. dataset_text_field="text":
    #    Tells the trainer that our dataset has a "text" column containing
    #    the formatted prompts (from data_loader.py). The trainer will
    #    automatically tokenize this column.
    #
    # 2. max_seq_length:
    #    Maximum sequence length for tokenization. Sequences longer than
    #    this are truncated. Shorter ones are padded.
    #
    # 3. packing=False:
    #    When True, multiple short examples are concatenated into one
    #    sequence to maximize GPU utilization. We disable this for
    #    simplicity and cleaner per-example loss computation.
    #    Enable if your examples are much shorter than max_seq_length.
    #
    # 4. peft_config is NOT passed here because we already applied PEFT
    #    to the model in model_setup.py. SFTTrainer detects the PEFT
    #    wrapper automatically.

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # Report memory before training
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"   GPU memory before training: {mem:.2f} / {total:.1f} GB")

    # ==================================================================
    # Step 6: TRAIN!
    # ==================================================================
    print("\n" + "=" * 60)
    print("  🏁 STARTING TRAINING")
    print("=" * 60)
    print("   This will take a while depending on your GPU and dataset size.")
    print("   You'll see loss values being printed — they should decrease over time.")
    print("   A typical training run on 5000 examples with a single GPU takes ~30-60 min.")
    print()

    # Enable torch anomaly detection for debugging (comment out for production)
    # torch.autograd.set_detect_anomaly(True)

    try:
        # THE ACTUAL TRAINING CALL
        # This is where the magic happens. trainer.train() runs the loop:
        #   for epoch in epochs:
        #       for batch in train_dataloader:
        #           loss = model(batch)     ← Forward pass
        #           loss.backward()          ← Backward pass
        #           optimizer.step()         ← Update weights
        #           scheduler.step()         ← Update learning rate
        #           if step % log_steps: log()
        #           if step % save_steps: save_checkpoint()
        #           if step % eval_steps: evaluate()
        train_result = trainer.train()

    except torch.cuda.OutOfMemoryError:
        print("\n" + "❌" * 30)
        print("  CUDA OUT OF MEMORY!")
        print("❌" * 30)
        print("\n  Your GPU doesn't have enough VRAM. Try these fixes:")
        print("  1. Reduce batch size:   per_device_train_batch_size = 1")
        print("  2. Reduce seq length:   max_seq_length = 256")
        print("  3. Increase grad accum: gradient_accumulation_steps = 8")
        print("  4. Disable compile:     use_torch_compile = False")
        print("  5. Enable grad checkpoint: gradient_checkpointing = True (default)")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\nCommon causes and fixes:")
        print("  • 'Token not found' → Set HF_TOKEN environment variable")
        print("  • 'Triton error'    → Set use_torch_compile = False")
        print("  • 'NaN loss'        → Reduce learning_rate to 5e-5")
        raise

    # ==================================================================
    # Step 7: Save the model
    # ==================================================================
    print("\n" + "=" * 60)
    print("  💾 SAVING MODEL")
    print("=" * 60)

    # Save the LoRA adapter weights (NOT the full base model)
    # The adapter is tiny (~10 MB) compared to the base model (~5 GB)
    final_dir = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"   ✅ Model saved to: {final_dir}")
    print(f"   Adapter size: {sum(f.stat().st_size for f in __import__('pathlib').Path(final_dir).rglob('*') if f.is_file()) / (1024**2):.1f} MB")

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ==================================================================
    # Step 8: Final evaluation
    # ==================================================================
    print("\n" + "=" * 60)
    print("  📊 FINAL EVALUATION ON VALIDATION SET")
    print("=" * 60)

    eval_results = trainer.evaluate()
    print(f"   Validation loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "🎉" * 30)
    print("  TRAINING COMPLETE!")
    print("🎉" * 30)
    print(f"\n   Model saved to:     {final_dir}")
    print(f"   Training loss:      {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Validation loss:    {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"   Total train steps:  {metrics.get('train_steps', 'N/A')}")
    print(f"   Training runtime:   {metrics.get('train_runtime', 0):.0f} seconds")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"   Peak GPU memory:    {peak_mem:.2f} GB")

    print(f"\n   Next steps:")
    print(f"   1. Run inference:    python inference.py --model_dir {final_dir}")
    print(f"   2. Evaluate:         python evaluate.py --model_dir {final_dir}")
    print()

    return final_dir


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    final_dir = train()
