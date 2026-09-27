#!/usr/bin/env python
"""
09_merge_and_export.py — merge LoRA adapters into the base, verify, and export.

Why merge at all?
-----------------
An unmerged adapter is two sets of weights at inference: the base plus a low-rank
delta. Serving frameworks can either:
  (a) keep them separate and apply the delta per-request (vLLM/SGLang can hot-swap
      adapters this way — great when you serve many adapters over one base), or
  (b) merge once and serve a single dense model (simpler, slightly faster, and the
      only option for llama.cpp/Ollama and most edge runtimes).

Merging is lossy in one specific way that catches people out: **merging a LoRA that was
trained on a 4-bit (QLoRA) base into a full-precision base introduces an error.** The
adapter learned to correct for the quantization error of its own base. Merge it into a
different base and that correction is now wrong. If you trained QLoRA, either serve the
adapter on the same quantized base, or merge into the SAME quantized base and re-quantize.

Run it
------
    python 09_merge_and_export.py --base Qwen/Qwen2.5-7B-Instruct --adapter ./out/sft-lora --out ./out/merged
    python 09_merge_and_export.py --base ... --adapter ... --out ... --gguf Q4_K_M
    python 09_merge_and_export.py --base ... --adapter ... --out ... --push user/repo --verify
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  — UTF-8 console fix; see common/__init__.py

VERIFY_PROMPTS = [
    "Give me a one-sentence summary of what a transformer is.",
    "List three symptoms of dehydration.",
    "Reply with exactly this JSON and nothing else: {\"ok\": true}",
    "What is 17 * 23? Show your working.",
    "Write a haiku about databases.",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="Base model id or path")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--out", required=True, help="Output directory for the merged model")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--gguf", default=None, help="Also export GGUF with this quant type (e.g. Q4_K_M)")
    p.add_argument("--push", default=None, help="HF repo id to push to (created private)")
    p.add_argument("--verify", action="store_true", help="Run sanity generations on merged vs adapter")
    p.add_argument("--device", default="cpu", help="cpu is usually safest for merging")
    a = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]

    print(f"  loading base       {a.base}")
    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=dtype, device_map=a.device, trust_remote_code=True)
    print(f"  loading adapter    {a.adapter}")
    model = PeftModel.from_pretrained(base, a.adapter)

    # Sanity: confirm the adapter actually targets layers that exist in this base.
    cfg_path = Path(a.adapter) / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        targets = cfg.get("target_modules", [])
        linear = {n.split(".")[-1] for n, m in base.named_modules()
                  if m.__class__.__name__ == "Linear"}
        missing = [t for t in targets if t not in linear]
        print(f"  adapter r/alpha    {cfg.get('r')} / {cfg.get('lora_alpha')}")
        print(f"  target modules     {targets}")
        if missing:
            print(f"  ⚠  target modules not found in this base: {missing}")
            print("     Merging may silently no-op for those layers. Check you used the")
            print("     right base model — this is the #1 merge bug.")
        if cfg.get("base_model_name_or_path") and \
                cfg["base_model_name_or_path"].split("/")[-1] != a.base.split("/")[-1]:
            print(f"  ⚠  adapter was trained on '{cfg['base_model_name_or_path']}' but you "
                  f"are merging into '{a.base}'. Deltas are only meaningful for the base "
                  f"they were trained on.")

    print("  merging (this is the point of no return for the base weights)...")
    merged = model.merge_and_unload()

    merged.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    tok.save_pretrained(str(out))
    print(f"  ✅ merged model saved to {out}")

    # Record provenance — months later this file is the only way to know what you shipped.
    (out / "MERGE_CARD.json").write_text(json.dumps({
        "base_model": a.base,
        "adapter": str(a.adapter),
        "dtype": a.dtype,
        "merge_method": "peft.merge_and_unload",
        "warning": ("If the adapter was trained with QLoRA on a 4-bit base, merging into "
                    "a full-precision base changes behaviour. Re-verify."),
    }, indent=2))
    print(f"  wrote {out / 'MERGE_CARD.json'}")

    if a.verify:
        _verify(a.base, a.adapter, str(out))

    if a.gguf:
        _export_gguf(str(out), a.gguf)

    if a.push:
        merged.push_to_hub(a.push, private=True)
        tok.push_to_hub(a.push, private=True)
        print(f"  ✅ pushed to https://huggingface.co/{a.push}")


def _verify(base_id: str, adapter_id: str, merged_path: str) -> None:
    """
    Compare merged vs adapter-on-base. They should agree closely. If they diverge on
    obvious prompts, the merge was wrong (wrong base, wrong target modules, or a
    QLoRA-into-fp16 mismatch).

    This is cheap and catches the entire class of silent merge failures, which are
    otherwise only discovered in production.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n  ── verifying merge ──")
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # bf16, not fp16 — and that choice is load-bearing for this comparison.
    # CH-06 §5.5: merging in fp16 loses precision in the W + BA sum, so an fp16
    # round-trip can make the merged model measurably worse than the adapter it came
    # from. Verifying in fp16 would therefore compare two models that have BOTH been
    # degraded by the dtype, and would hide exactly the defect this function exists to
    # catch. bf16 matches how the artifact is served.
    dtype = torch.bfloat16
    print("  loading merged model...")
    m1 = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=dtype,
                                              device_map="cpu")
    print("  loading base+adapter...")
    m2 = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype,
                                             device_map="cpu"), adapter_id)

    def gen(model, prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # add_special_tokens=False: apply_chat_template already inserted them. Without
        # this the prompt gets a second BOS (CH-06 §1.3 row 4, §13 error table), and
        # since both models see the same malformed prompt the comparison still "passes"
        # while neither side is being asked what you think it is.
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    agree = 0
    for q in VERIFY_PROMPTS:
        r1, r2 = gen(m1, q), gen(m2, q)
        same = r1[:80] == r2[:80]
        agree += same
        print(f"\n  Q: {q}")
        print(f"    merged : {r1[:110]!r}")
        print(f"    adapter: {r2[:110]!r}")
        if not same:
            print("    ⚠  outputs diverge")

    print(f"\n  agreement: {agree}/{len(VERIFY_PROMPTS)}")
    if agree < len(VERIFY_PROMPTS):
        print("  ⚠  Divergence is expected in exact tokens but the gist should match.")
        print("     If they are completely different, the merge is broken: check the base")
        print("     model id in adapter_config.json and the target_modules warning above.")


def _export_gguf(model_path: str, quant_type: str) -> None:
    print(f"\n  ── exporting GGUF ({quant_type}) ──")
    out_dir = Path(model_path).parent / f"gguf-{quant_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / "model-f16.gguf"

    c1 = [sys.executable, "-m", "llama_cpp.convert_hf_to_gguf",
          model_path, "--outfile", str(f16), "--outtype", "f16"]
    if subprocess.run(c1).returncode != 0:
        print("  ⚠  conversion failed — install llama.cpp tooling:")
        print("       pip install llama-cpp-python")
        print("     or build https://github.com/ggerganov/llama.cpp and use ./convert_hf_to_gguf.py")
        return

    c2 = [sys.executable, "-m", "llama_cpp.llama_quant",
          str(f16), str(out_dir / f"model-{quant_type}.gguf"), quant_type]
    subprocess.run(c2, check=False)
    f16.unlink(missing_ok=True)

    print(f"  ✅ GGUF at {out_dir}")
    print("\n  ⚠  The chat template must be embedded in the GGUF, or the model will be")
    print("     served with the wrong prompt format and quality will silently degrade.")
    print("     Verify with:  llama-cli -m <file> -p 'Hello' --chat-template <name>")


if __name__ == "__main__":
    main()
