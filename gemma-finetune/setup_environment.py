"""
============================================================================
SETUP_ENVIRONMENT.PY — Pre-Flight Environment Checks
============================================================================

PURPOSE:
    Run this script BEFORE training to verify your environment is ready.
    It checks Python version, GPU, CUDA, and all required packages.

USAGE:
    python setup_environment.py

WHAT IT CHECKS:
    1. Python version (>= 3.10 required)
    2. CUDA availability and version
    3. GPU memory (warns if < 8 GB)
    4. All required Python packages
    5. bitsandbytes CUDA compatibility
    6. torch.compile / Triton availability
    7. Hugging Face authentication token

EXIT CODES:
    0 — All checks passed (green light to train!)
    1 — Critical issue found (must fix before training)
"""

import sys
import os
import importlib
import subprocess


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_pass(message: str) -> None:
    """Print a green PASS message."""
    print(f"  ✅ PASS: {message}")


def print_warn(message: str) -> None:
    """Print a yellow WARNING message."""
    print(f"  ⚠️  WARN: {message}")


def print_fail(message: str) -> None:
    """Print a red FAIL message."""
    print(f"  ❌ FAIL: {message}")


def check_python_version() -> bool:
    """
    Check 1: Python version.
    
    WHY >= 3.10:
      • dataclasses with slots (Python 3.10+)
      • Pattern matching syntax support
      • torch.compile requires Python >= 3.8, but 3.10+ is recommended
      • Newer type hint syntax (X | Y instead of Union[X, Y])
    """
    print_header("CHECK 1: Python Version")

    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"

    if major < 3 or (major == 3 and minor < 10):
        print_fail(f"Python {version_str} detected. Need >= 3.10")
        print(f"         Install Python 3.10+: https://www.python.org/downloads/")
        return False
    else:
        print_pass(f"Python {version_str}")
        return True


def check_cuda() -> bool:
    """
    Check 2: CUDA and GPU availability.
    
    WHY CUDA:
      • Fine-tuning a 2B model on CPU would take weeks instead of hours
      • bitsandbytes 4-bit quantization ONLY works on NVIDIA GPUs
      • torch.compile's "inductor" backend requires CUDA for GPU kernel generation
      
    COMMON ISSUES:
      • "CUDA not available" 
          → Install NVIDIA drivers: https://developer.nvidia.com/cuda-downloads
          → Make sure you installed the CUDA version of PyTorch, not CPU
          → Run: pip install torch --index-url https://download.pytorch.org/whl/cu121
      
      • "CUDA version mismatch"
          → Your PyTorch was built for a different CUDA version
          → Check: nvidia-smi (shows driver CUDA) vs torch.version.cuda (PyTorch CUDA)
          → The driver CUDA must be >= PyTorch's CUDA version
    """
    print_header("CHECK 2: CUDA & GPU")

    try:
        import torch

        if not torch.cuda.is_available():
            print_fail("CUDA is NOT available!")
            print("         Possible causes:")
            print("         1. No NVIDIA GPU installed")
            print("         2. NVIDIA drivers not installed")
            print("         3. PyTorch was installed without CUDA support")
            print("         Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False

        # CUDA is available
        cuda_version = torch.version.cuda
        print_pass(f"CUDA {cuda_version} available")

        # Check GPU details
        gpu_count = torch.cuda.device_count()
        print_pass(f"{gpu_count} GPU(s) detected:")

        all_ok = True
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
            capability = torch.cuda.get_device_capability(i)
            print(f"         GPU {i}: {name}")
            print(f"                Memory: {mem_gb:.1f} GB")
            print(f"                Compute Capability: {capability[0]}.{capability[1]}")

            if mem_gb < 8:
                print_warn(f"GPU {i} has only {mem_gb:.1f} GB VRAM. "
                           f"Minimum 8 GB recommended for QLoRA.")
                print("         → Reduce batch_size to 1 and max_seq_length to 256")
                all_ok = False

            if capability[0] >= 8:
                print_pass(f"GPU {i} supports bfloat16 (Ampere+)")
            else:
                print_warn(f"GPU {i} does NOT support bfloat16. Use float16 instead.")
                print("         → In config.py: set bf16=False, fp16=True, "
                      "bnb_4bit_compute_dtype='float16'")

        return all_ok

    except ImportError:
        print_fail("PyTorch is not installed!")
        print("         Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_required_packages() -> bool:
    """
    Check 3: All required Python packages.
    
    Each package is checked for importability and version.
    """
    print_header("CHECK 3: Required Packages")

    # (module_name, package_name_for_pip, minimum_version_or_None)
    required = [
        ("torch",           "torch",          "2.2.0"),
        ("transformers",    "transformers",   "4.38.0"),
        ("peft",            "peft",           "0.8.0"),
        ("bitsandbytes",    "bitsandbytes",   "0.42.0"),
        ("datasets",        "datasets",       "2.16.0"),
        ("accelerate",      "accelerate",     "0.27.0"),
        ("trl",             "trl",            "0.7.0"),
        ("sentencepiece",   "sentencepiece",  None),
        ("google.protobuf", "protobuf",       None),
        ("rouge_score",     "rouge-score",    None),
        ("nltk",            "nltk",           None),
    ]

    all_ok = True
    missing = []

    for module_name, pip_name, min_version in required:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")

            if min_version and version != "unknown":
                from packaging.version import Version

                try:
                    if Version(version) < Version(min_version):
                        print_warn(f"{pip_name} {version} < {min_version} (upgrade recommended)")
                    else:
                        print_pass(f"{pip_name} {version}")
                except Exception:
                    print_pass(f"{pip_name} {version} (version check skipped)")
            else:
                print_pass(f"{pip_name} {version}")

        except ImportError:
            print_fail(f"{pip_name} is NOT installed")
            print(f"         Run: pip install {pip_name}")
            missing.append(pip_name)
            all_ok = False

    if missing:
        print(f"\n  To install all missing packages at once:")
        print(f"    pip install {' '.join(missing)}")

    return all_ok


def check_bitsandbytes() -> bool:
    """
    Check 4: bitsandbytes CUDA compatibility.
    
    COMMON ISSUES:
      • "libcudart.so not found"
          → Set LD_LIBRARY_PATH:
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
      
      • "CUDA extension not available"
          → bitsandbytes was compiled for a different CUDA version
          → Reinstall: pip install bitsandbytes --force-reinstall
      
      • Windows issues:
          → Use WSL2, native Windows support is experimental
    """
    print_header("CHECK 4: bitsandbytes CUDA Integration")

    try:
        import bitsandbytes as bnb
        import torch

        if torch.cuda.is_available():
            # Try to create a 4-bit quantized linear layer as a smoke test
            try:
                linear = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16)
                print_pass("bitsandbytes 4-bit operations working")
                return True
            except Exception as e:
                print_fail(f"bitsandbytes CUDA error: {e}")
                print("         Try: pip install bitsandbytes --force-reinstall")
                print("         Also: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
                return False
        else:
            print_warn("Cannot test bitsandbytes (no CUDA). Will fail during training.")
            return False

    except ImportError:
        print_fail("bitsandbytes not installed")
        return False


def check_torch_compile() -> bool:
    """
    Check 5: torch.compile and Triton availability.
    
    torch.compile REQUIREMENTS:
      • PyTorch >= 2.0 (we need 2.2+)
      • Triton package for the "inductor" backend (Linux only)
      • CUDA-capable GPU
      • Python >= 3.8
    
    WHAT HAPPENS IF TRITON IS MISSING:
      You can still use torch.compile with the "eager" or "aot_eager" backends,
      but you won't get the GPU kernel optimization that makes it fast.
      Training will still work — just without the speedup.
    
    KNOWN torch.compile ISSUES:
      1. "Triton not found" → pip install triton
      2. "Graph break" messages → Normal with LoRA. Set fullgraph=False (our default).
      3. "Recompilation" warnings → The model is being recompiled due to shape changes.
         Set torch_compile_dynamic=True in config to handle dynamic shapes.
      4. Very slow first few steps → Normal. Compilation happens on first run.
         Subsequent steps will be fast.
      5. "CUDA out of memory during compilation" → torch.compile uses extra memory
         for optimization. Reduce batch_size or set use_torch_compile=False.
    """
    print_header("CHECK 5: torch.compile & Triton")

    try:
        import torch

        # Check torch.compile availability
        if hasattr(torch, "compile"):
            print_pass(f"torch.compile available (PyTorch {torch.__version__})")
        else:
            print_fail(f"torch.compile NOT available (PyTorch {torch.__version__} is too old)")
            print("         Upgrade: pip install torch>=2.2.0")
            return False

        # Check Triton
        try:
            import triton
            print_pass(f"Triton {triton.__version__} available (inductor backend ready)")
        except ImportError:
            print_warn("Triton is NOT installed (inductor backend will not work)")
            print("         Without Triton, torch.compile will use slower backends.")
            print("         Install: pip install triton")
            print("         Note: Triton is Linux-only. Not available on Windows/macOS.")

        # Test a simple compilation
        try:
            @torch.compile(backend="inductor")
            def test_fn(x):
                return x * 2 + 1

            if torch.cuda.is_available():
                x = torch.randn(4, 4, device="cuda")
                _ = test_fn(x)
                print_pass("torch.compile test passed with inductor backend")
            else:
                print_warn("Cannot test torch.compile on CUDA (no GPU)")

        except Exception as e:
            print_warn(f"torch.compile test failed: {e}")
            print("         torch.compile may still work during training.")
            print("         If it doesn't, set use_torch_compile=False in config.py.")

        return True

    except ImportError:
        print_fail("PyTorch not installed")
        return False


def check_hf_token() -> bool:
    """
    Check 6: Hugging Face authentication.
    
    WHY YOU NEED A TOKEN:
      Google's Gemma models require you to:
      1. Accept the license agreement on https://huggingface.co/google/gemma-2b
      2. Create an access token with "read" permissions
      3. Set it as an environment variable or login via CLI
    
    HOW TO SET UP:
      Option 1: Environment variable
        export HF_TOKEN=hf_your_token_here
      
      Option 2: CLI login (saves token to disk)
        huggingface-cli login
      
      Option 3: In Python
        from huggingface_hub import login
        login(token="hf_your_token_here")
    """
    print_header("CHECK 6: Hugging Face Token")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token:
        # Mask token for security
        masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
        print_pass(f"HF_TOKEN found: {masked}")
        return True
    else:
        # Check if token is saved via huggingface-cli login
        try:
            from huggingface_hub import HfFolder
            saved_token = HfFolder.get_token()
            if saved_token:
                print_pass("Hugging Face token found (saved via CLI login)")
                return True
        except Exception:
            pass

        print_warn("No Hugging Face token found!")
        print("         Gemma requires authentication. Set your token:")
        print("           export HF_TOKEN=hf_your_token_here")
        print("         Or login via CLI:")
        print("           huggingface-cli login")
        print("         Get your token: https://huggingface.co/settings/tokens")
        return False


def main():
    """Run all environment checks and report results."""
    print("\n" + "🔍" * 30)
    print("  GEMMA FINE-TUNING — ENVIRONMENT CHECK")
    print("🔍" * 30)

    results = {
        "Python Version": check_python_version(),
        "CUDA & GPU": check_cuda(),
        "Required Packages": check_required_packages(),
        "bitsandbytes": check_bitsandbytes(),
        "torch.compile": check_torch_compile(),
        "HF Token": check_hf_token(),
    }

    # Summary
    print_header("SUMMARY")
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL/WARN"
        print(f"  {check_name:30s} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  🎉 All checks passed! You're ready to fine-tune Gemma!")
        print("     Run: python train.py\n")
        sys.exit(0)
    else:
        print("\n  ⚠️  Some checks failed. Please fix the issues above.")
        print("     Training may still work, but you might encounter errors.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
