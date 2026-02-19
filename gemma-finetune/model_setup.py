"""
============================================================================
MODEL_SETUP.PY — Gemma Model Loading with QLoRA Configuration
============================================================================

PURPOSE:
    Loads the Gemma model from Hugging Face Hub with 4-bit quantization
    (QLoRA) and attaches LoRA adapter layers for parameter-efficient
    fine-tuning.

WHAT THIS FILE DOES:
    1. Configures 4-bit quantization via BitsAndBytesConfig
    2. Loads the Gemma model with quantization applied
    3. Prepares the model for k-bit training
    4. Applies LoRA adapters via PEFT
    5. Optionally applies torch.compile() for speedup
    6. Reports memory usage and trainable parameter count

THE BIG PICTURE:
    Without QLoRA:
      Gemma-2B in fp16 needs ~4 GB just for weights.
      Training needs ~3x that (gradients + optimizer) = ~12 GB.
      
    With QLoRA:
      Weights compressed to 4-bit = ~1 GB.
      Only LoRA params (~2.6M) need gradients/optimizer = ~0.5 GB.
      Total: ~6 GB VRAM. Fits on consumer GPUs!

USAGE:
    from model_setup import load_model_and_tokenizer
    from config import TrainingConfig

    cfg = TrainingConfig()
    model, tokenizer = load_model_and_tokenizer(cfg)
"""

import os
import sys
import warnings
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable vs. total parameters.
    
    This shows the dramatic efficiency of LoRA:
    - Total params: ~2 billion (frozen, not updated during training)
    - Trainable params: ~2.6 million (LoRA adapters, updated during training)
    - Trainable %: ~0.13%
    
    This is why LoRA is called "Parameter-Efficient Fine-Tuning" —
    we're training less than 1% of the model's parameters!
    """
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    trainable_pct = 100 * trainable / total if total > 0 else 0

    print(f"\n📊 Model Parameter Summary:")
    print(f"   Total parameters:     {total:>15,}")
    print(f"   Trainable parameters: {trainable:>15,}")
    print(f"   Trainable %:          {trainable_pct:>14.4f}%")
    print(f"   Frozen parameters:    {total - trainable:>15,}")


def get_quantization_config(config) -> BitsAndBytesConfig:
    """
    Create the BitsAndBytesConfig for 4-bit quantization.
    
    WHAT EACH PARAMETER DOES:
    
    load_in_4bit:
        When True, each model weight is stored using only 4 bits instead of
        16 bits. This compresses the model by ~4x.
        
        How 4-bit storage works:
        - Weights are grouped into blocks of 64
        - Each block has a shared scaling factor (stored in fp16)
        - Each weight in the block is quantized to one of 16 values (4 bits)
        - During computation, weights are dequantized back to the compute dtype
        
    bnb_4bit_quant_type:
        The algorithm used to choose those 16 quantization values.
        
        "nf4" (Normal Float 4):
        - Creates 16 bins optimally spaced for normally distributed data
        - Neural network weights approximately follow a normal distribution
        - This gives the best quantization accuracy
        
        "fp4" (Float Point 4):
        - Uses standard floating-point representation with 4 bits
        - Slightly faster to quantize but lower quality
        
    bnb_4bit_compute_dtype:
        Even though weights are STORED in 4-bit, all math is done in this dtype.
        Before each matrix multiplication, the 4-bit weights are upcast to this type.
        
        bfloat16: Best accuracy, requires Ampere+ GPU (RTX 3000+, A100)
        float16:  Good accuracy, works on older GPUs
        float32:  Maximum accuracy but 2x slower
        
    bnb_4bit_use_double_quant:
        "Double quantization" — quantizes the quantization scaling factors too.
        
        Without double quant: Each block of 64 weights has a fp16 scaling factor = 2 bytes
        With double quant: Those scaling factors are also quantized to 8-bit = 1 byte
        
        Net memory savings: ~0.4 GB for Gemma-2B. Free lunch!
    
    Returns:
        Configured BitsAndBytesConfig
    """
    if not config.use_4bit:
        print("   ℹ️  4-bit quantization DISABLED. Using full precision.")
        return None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=config.compute_dtype,
        bnb_4bit_use_double_quant=config.use_double_quant,
    )

    print(f"   Quantization: 4-bit ({config.bnb_4bit_quant_type})")
    print(f"   Compute dtype: {config.bnb_4bit_compute_dtype}")
    print(f"   Double quantization: {config.use_double_quant}")

    return bnb_config


def get_lora_config(config) -> LoraConfig:
    """
    Create the LoRA adapter configuration.
    
    LoRA EXPLAINED IN DETAIL:
    
    Standard fine-tuning updates the full weight matrix W (shape: d×d).
    For Gemma-2B, d=2048, so W has 4M parameters per layer.
    With 18 layers × 4 attention matrices = 288M parameters to update.
    
    LoRA instead decomposes the update as:
        W_new = W_original + (alpha/r) × A × B
    
    Where:
        W_original: Original frozen weight (d×d) — NOT updated
        A: Trainable down-projection (d×r) — TINY
        B: Trainable up-projection (r×d) — TINY
        r: LoRA rank (e.g., 16) — much smaller than d (2048)
        alpha: Scaling factor
    
    Parameter count per adapted layer:
        Original: d × d = 2048 × 2048 = 4,194,304
        LoRA:     d × r + r × d = 2048 × 16 + 16 × 2048 = 65,536
        Ratio:    65,536 / 4,194,304 = 1.56% per layer
    
    After training, A×B can be MERGED into W_original:
        W_final = W_original + A × B
    This means zero inference overhead — the model runs at full speed!
    
    PARAMETERS EXPLAINED:
    
    r (rank):
        Controls adapter capacity. Higher = more expressive but more params.
        Think of it as the "bottleneck dimension" of the adapter.
        
    lora_alpha:
        Scaling factor: output = base + (alpha/r) × adapter.
        Higher alpha = adapter has more influence on the output.
        Rule of thumb: alpha = 2 × r.
        
    lora_dropout:
        Random dropout applied to adapter outputs during training.
        Prevents overfitting. 0.05 = 5% of values zeroed each step.
        
    target_modules:
        Which linear layers get LoRA adapters. In Gemma's transformer:
        - q_proj: Query matrix in self-attention
        - k_proj: Key matrix in self-attention  
        - v_proj: Value matrix in self-attention
        - o_proj: Output projection in self-attention
        - gate_proj, up_proj, down_proj: MLP layers (optional, more params)
        
    task_type:
        CAUSAL_LM = autoregressive language modeling (next token prediction).
        This is what Gemma does — it predicts one token at a time.
        
    bias:
        Whether to train bias terms in LoRA layers.
        "none" = don't train biases (saves a tiny bit of memory).
        
    Returns:
        Configured LoraConfig
    """
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    print(f"\n🔧 LoRA Configuration:")
    print(f"   Rank (r):         {config.lora_r}")
    print(f"   Alpha:            {config.lora_alpha}")
    print(f"   Scaling factor:   {config.lora_alpha / config.lora_r:.2f}x")
    print(f"   Dropout:          {config.lora_dropout}")
    print(f"   Target modules:   {config.target_modules}")
    print(f"   Bias:             none")

    return lora_config


def apply_torch_compile(model, config):
    """
    Optionally apply torch.compile() to the model.
    
    torch.compile() DEEP DIVE:
    
    WHAT IT DOES:
        PyTorch normally runs in "eager mode" — each operation (matmul, relu,
        etc.) is executed immediately as Python encounters it. This has overhead
        from Python's interpreter and prevents cross-operation optimization.
        
        torch.compile() traces the model's forward pass, builds a computation
        graph, and compiles it into optimized GPU code. This means:
        - Operations are fused (e.g., matmul + bias + relu → one kernel)
        - Memory access patterns are optimized
        - Unnecessary intermediate tensors are eliminated
        
    BACKENDS:
        "inductor" (default):
            - Uses OpenAI Triton to generate custom GPU kernels
            - Best performance for most models
            - Requires the 'triton' Python package
            - Linux only
            
        "cudagraphs":
            - Captures the entire CUDA call sequence and replays it
            - Very fast for fixed-shape inputs
            - Less flexible (breaks if tensor shapes change)
            
        "eager":
            - No compilation. Used for debugging.
            - torch.compile with eager = just adds tracing overhead
            
        "aot_eager":
            - Ahead-of-time tracing + eager execution
            - Useful for debugging graph breaks
    
    MODES:
        "default":
            - Balanced compilation time and runtime performance
            - Good for most use cases
            
        "reduce-overhead":
            - Uses CUDA graphs to eliminate Python overhead
            - Better for small models / fast iterations
            - May use more memory
            
        "max-autotune":
            - Tries many kernel configurations, picks the fastest
            - Compilation is VERY slow (10-30 minutes)
            - Best runtime performance
            - Only use for final production training runs
    
    COMMON ISSUES AND SOLUTIONS:
    
    Issue 1: "Cannot find triton"
        → pip install triton
        → Triton is Linux-only. On Windows: set use_torch_compile=False
        
    Issue 2: "Graph break in LoRA forward"
        → Normal! LoRA's dynamic adapter switching causes graph breaks
        → Our default fullgraph=False handles this gracefully
        → The compiled parts are still faster; uncompiled parts run in eager
        
    Issue 3: "CUDA out of memory during compilation"
        → torch.compile needs extra memory for graph analysis
        → Reduce batch_size by 1-2, or disable torch.compile
        
    Issue 4: "Recompilation triggered"
        → Happens when input shapes change (different sequence lengths)
        → Set torch_compile_dynamic=True to handle variable shapes
        → Or ensure all inputs are padded to the same length
        
    Issue 5: "Slow first N steps"
        → NORMAL. Compilation happens on the first run.
        → First 3-5 steps may take 1-5 minutes each.
        → All subsequent steps will be much faster.
        
    Issue 6: "torch._dynamo errors"
        → Try: import torch._dynamo; torch._dynamo.reset()
        → Or downgrade to a different PyTorch version
    
    Args:
        model: The loaded model (with LoRA adapters applied)
        config: TrainingConfig instance
        
    Returns:
        The model (possibly compiled)
    """
    if not config.use_torch_compile:
        print("\n⚡ torch.compile: DISABLED")
        print("   Model will run in eager mode (no compilation speedup).")
        return model

    print(f"\n⚡ Applying torch.compile()...")
    print(f"   Backend:    {config.torch_compile_backend}")
    print(f"   Mode:       {config.torch_compile_mode}")
    print(f"   Full graph: {config.torch_compile_fullgraph}")
    print(f"   Dynamic:    {config.torch_compile_dynamic}")

    try:
        # Check if Triton is available (needed for inductor backend)
        if config.torch_compile_backend == "inductor":
            try:
                import triton  # noqa: F401
                print(f"   Triton:     ✅ Available (v{triton.__version__})")
            except ImportError:
                print("   Triton:     ❌ Not installed!")
                print("                  Falling back to 'eager' backend.")
                print("                  Install Triton for speedup: pip install triton")
                config.torch_compile_backend = "eager"

        # Apply torch.compile
        model = torch.compile(
            model,
            backend=config.torch_compile_backend,
            mode=config.torch_compile_mode,
            fullgraph=config.torch_compile_fullgraph,
            dynamic=config.torch_compile_dynamic,
        )

        print("   ✅ torch.compile applied successfully!")
        print("   ⏳ Note: First few training steps will be slow (compilation).")
        print("          Subsequent steps will be 1.3-2x faster.")

    except Exception as e:
        print(f"   ❌ torch.compile FAILED: {e}")
        print("      Continuing without compilation (eager mode).")
        print("      Training will still work, just without speedup.")
        print("      To fix: try a different backend or set use_torch_compile=False")

    return model


def load_model_and_tokenizer(config) -> Tuple:
    """
    Main function: Load Gemma with QLoRA and LoRA adapters.
    
    COMPLETE FLOW:
    
    1. Create quantization config (how to compress weights)
    2. Load pre-trained Gemma model from Hugging Face Hub
    3. Prepare for k-bit training (fix batch normalization, etc.)
    4. Apply LoRA adapters (add trainable layers)
    5. Optionally apply torch.compile (optimize computation graph)
    6. Load tokenizer
    
    MEMORY USAGE BREAKDOWN (Gemma-2B with QLoRA):
        Model weights (4-bit):     ~1.0 GB
        LoRA adapters (fp16):      ~0.01 GB
        Optimizer states:          ~0.5 GB
        Gradients:                 ~0.5 GB
        Activations (batch=4):     ~2.0 GB
        CUDA overhead:             ~1.0 GB
        ─────────────────────────────────
        TOTAL:                     ~5.0 GB
        
    With gradient checkpointing: ~4.0 GB (saves ~1 GB activations)
    
    Args:
        config: TrainingConfig instance
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print("\n" + "=" * 60)
    print("  MODEL SETUP: Loading Gemma with QLoRA")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Quantization config
    # ------------------------------------------------------------------
    print("\n📦 Step 1: Configuring 4-bit quantization...")
    bnb_config = get_quantization_config(config)

    # ------------------------------------------------------------------
    # Step 2: Load the base model
    # ------------------------------------------------------------------
    print(f"\n🤖 Step 2: Loading {config.model_name}...")
    print(f"   This may take 1-2 minutes on first run (downloading ~5 GB)...")
    print(f"   Subsequent runs use the cached version in ~/.cache/huggingface/")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,

        # Quantization config
        quantization_config=bnb_config,

        # Device placement
        # "auto" = distribute across available GPUs, spill to CPU if needed
        device_map="auto",

        # Trust remote code (False for Gemma — it's natively supported)
        trust_remote_code=config.trust_remote_code,

        # Attention implementation
        # "eager" = standard attention (most compatible)
        # "sdpa" = PyTorch's scaled dot-product attention (faster, built-in)
        # "flash_attention_2" = fastest, but needs flash-attn package
        attn_implementation="eager",

        # Use auth token from environment variable
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"   ✅ Model loaded!")
    print(f"   Architecture: {model.config.model_type}")
    print(f"   Hidden size: {model.config.hidden_size}")
    print(f"   Num layers: {model.config.num_hidden_layers}")
    print(f"   Num attention heads: {model.config.num_attention_heads}")

    # Report GPU memory after loading
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"   GPU memory allocated: {mem_allocated:.2f} GB")
        print(f"   GPU memory reserved:  {mem_reserved:.2f} GB")

    # ------------------------------------------------------------------
    # Step 3: Prepare for k-bit training
    # ------------------------------------------------------------------
    print("\n🔧 Step 3: Preparing model for k-bit training...")

    # This function does several important things:
    # 1. Casts layer norms to float32 (they break in low precision)
    # 2. Casts the output head (lm_head) to float32
    # 3. Enables input gradient computation (needed for backprop through quantized layers)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )

    if config.gradient_checkpointing:
        print("   ✅ Gradient checkpointing ENABLED (saves ~30-50% VRAM)")
        # Enable gradient checkpointing with a kwarg fix for newer HF versions
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        print("   ℹ️  Gradient checkpointing DISABLED")

    # ------------------------------------------------------------------
    # Step 4: Apply LoRA adapters
    # ------------------------------------------------------------------
    print("\n🔌 Step 4: Applying LoRA adapters...")
    lora_config = get_lora_config(config)

    # This wraps the base model with LoRA adapter layers
    # Only the adapter parameters will be updated during training
    model = get_peft_model(model, lora_config)

    # Show parameter counts
    print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # Step 5: Apply torch.compile (optional)
    # ------------------------------------------------------------------
    model = apply_torch_compile(model, config)

    # ------------------------------------------------------------------
    # Step 6: Load tokenizer
    # ------------------------------------------------------------------
    print("\n📝 Step 6: Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
    )

    # Set padding token (required for batching during training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # Right-pad for training

    print(f"   ✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ✅ MODEL SETUP COMPLETE!")
    print("=" * 60)

    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"   GPU memory used:      {mem_allocated:.2f} / {mem_total:.1f} GB")
        print(f"   GPU memory available: {mem_total - mem_reserved:.2f} GB")

    return model, tokenizer


# ============================================================================
# Quick test: Run this file directly to test model loading
# ============================================================================
if __name__ == "__main__":
    from config import TrainingConfig

    cfg = TrainingConfig()
    print(cfg)

    model, tokenizer = load_model_and_tokenizer(cfg)

    print("\n🧪 Quick sanity check — generating tokens...")
    test_input = "Hello, I am Gemma. I can help you with"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Input:  {test_input}")
        print(f"   Output: {generated}")

    print("\n✅ Model loaded and generating successfully!")
