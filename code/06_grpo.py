#!/usr/bin/env python
"""
06_grpo.py — GRPO (Group Relative Policy Optimization) with a verifiable reward.

What it is
----------
GRPO is PPO without the value model (the "critic"). Instead of learning a baseline
to estimate the advantage, it samples a GROUP of G completions for the same prompt,
scores each with a reward function, and uses the group's own mean and standard
deviation as the baseline:

    A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)

That normalization is the entire idea. It removes the need for a learned value head,
which is what makes GRPO so much cheaper than PPO — one fewer model in memory, and no
value-loss to tune.

Why this matters in 2026
------------------------
GRPO became the default for **verifiable-reward** training (RLVR): math, code, format
compliance, tool-call validity — anything where you can *check* the answer instead of
asking a human or a reward model. DeepSeek-R1 popularized it. The insight is that when
the reward is a program rather than a learned approximation, you cannot reward-hack it
in the usual way, and RL becomes dramatically more reliable.

When NOT to use it
------------------
  ❌ Subjective qualities (tone, helpfulness, style). There is no verifier. Use DPO/ORPO.
  ❌ You cannot afford to generate G completions per prompt (G=8 is typical; that is
     8x the generation cost of DPO, and generation is the expensive part).
  ❌ Your model cannot produce a correct answer even by chance. If pass@G = 0 for every
     prompt, every advantage is 0 and nothing is learned. This is the #1 reason GRPO
     runs silently do nothing — check pass rate FIRST.

Run it
------
    python 06_grpo.py --dry-run --data data/math_prompts.jsonl
    python 06_grpo.py --data data/math_prompts.jsonl --num-generations 8 --reward math
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common.memory import TrainPlan, _print_plan       # noqa: E402

DEFAULTS = dict(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    data="data/math_prompts.jsonl",
    output="./out/grpo",
    reward="math",           # math | format | code
    num_generations=8,       # G: completions sampled per prompt. The memory/compute knob.
    max_prompt_len=256,
    max_completion_len=512,
    temperature=0.9,         # must be > 0. You need DIVERSITY within the group or
                             # std(r) = 0 and the advantage is undefined.
    top_p=0.95,
    lr=1e-6,                 # RL wants a *very* small LR. Raising this destabilises fast.
    epochs=1,
    batch=1,
    grad_accum=4,
    num_iterations=1,        # inner optimisation passes per batch of experience
    beta=0.04,               # KL coefficient to the reference policy
    lora_r=16,
    quant="4bit",
    seed=42,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, default=v, action=argparse.BooleanOptionalAction)
        elif isinstance(v, int):
            p.add_argument(flag, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(flag, type=float, default=v)
        else:
            p.add_argument(flag, default=v)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ======================================================================================
# REWARD FUNCTIONS — the heart of GRPO. A reward must be *programmatically checkable*.
# ======================================================================================
def extract_boxed(text: str) -> str | None:
    """Pull the answer out of \\boxed{...} — the standard math-answer convention."""
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m[-1].strip() if m else None


def extract_last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def reward_math(completions: list[str], answer: str, **_) -> list[float]:
    """
    Reward for a math problem. Partial credit is deliberate:

      1.0  correct final answer, well-formed
      0.3  correct final answer but no visible reasoning (we want CoT)
      0.1  produced a boxed answer but wrong (shaped reward: encourages the format)
      0.0  nothing usable

    Shaping the reward like this massively improves sample efficiency early in
    training, because the model gets a gradient signal before it can ever be right.
    The risk is that the model learns to emit boxed answers without solving anything —
    which is why the 1.0 tier requires correctness and the 0.3 tier exists to keep
    reasoning alive.
    """
    import math

    def norm(s: str) -> str:
        s = s.strip().replace(" ", "").replace("\\,", "")
        try:
            return f"{float(s):.6g}"
        except ValueError:
            return s.lower()

    rewards = []
    gold = norm(str(answer))
    for c in completions:
        pred = extract_boxed(c) or extract_last_number(c)
        has_reasoning = len(c) > 80 and ("=" in c or "therefore" in c.lower() or "so " in c.lower())
        if pred is None:
            rewards.append(0.0)
        elif norm(pred) == gold:
            rewards.append(1.0 if has_reasoning else 0.3)
        else:
            rewards.append(0.1)
    return rewards


def reward_format(completions: list[str], must_contain: list[str] | None = None,
                  must_match: str | None = None, **_) -> list[float]:
    """
    Reward for structural compliance — JSON validity, required tags, schema.

    This is the most under-used reward in practice. If your application needs a
    specific output shape, training on `reward_format` converges far faster and more
    reliably than any amount of SFT on format alone.
    """
    rewards = []
    must_contain = must_contain or []
    for c in completions:
        r = 0.0
        if must_match and re.search(must_match, c, re.S):
            r += 0.5
        if must_contain:
            r += 0.5 * sum(1 for m in must_contain if m in c) / len(must_contain)
        if c.strip().startswith("```json") or c.strip().startswith("{"):
            try:
                body = re.sub(r"^```(?:json)?|```$", "", c.strip(), flags=re.M)
                json.loads(body)
                r += 0.5
            except Exception:                       # noqa: BLE001
                pass
        rewards.append(min(r, 1.0))
    return rewards


def make_reward_fn(name: str):
    return {"math": reward_math, "format": reward_format}.get(name, reward_math)


# ======================================================================================
def main() -> None:
    a = parse_args()

    if not Path(a.data).exists():
        sys.exit(f"Not found: {a.data}\n"
                 "Expected JSONL with rows like:  {\"prompt\": \"...\", \"answer\": \"42\"}\n"
                 "Generate some:  python data/make_instruction_data.py --math --out data/math_prompts.jsonl")

    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  prompts            {len(rows):,}")
    if "answer" not in rows[0] and a.reward == "math":
        sys.exit("Reward 'math' needs an 'answer' field in every row. Found columns: "
                 f"{list(rows[0])}")

    # ----------------------------------------------------------------------------------
    # THE CRITICAL PRE-FLIGHT CHECK that almost nobody runs.
    # ----------------------------------------------------------------------------------
    print(f"\n  ── pre-flight ──")
    print(f"  reward function    {a.reward}")
    print(f"  group size G       {a.num_generations}")
    print(f"  samples per step   {a.num_generations * a.batch * a.grad_accum}")
    print("\n  ⚠  BEFORE running: verify your model can solve the task AT ALL.")
    print("     If pass@G is ~0 for every prompt, every group has std=0, every advantage")
    print("     is 0, and GRPO will run for hours learning nothing. Test with:")
    print(f"       python 06_grpo.py --data {a.data} --probe --num-generations 8")
    print("     Rule of thumb: you want a pass rate between roughly 5% and 80%.")
    print("     Below 5%, the task is too hard — use easier data or a bigger model.")
    print("     Above 80%, the task is too easy — the model gains nothing.")

    plan = TrainPlan(
        model=next((m for m in ["1.5B", "1B", "3B", "7B", "8B"] if m.lower() in a.model.lower()), "1.5B"),
        method="qlora" if a.quant == "4bit" else "lora",
        seq_len=a.max_prompt_len + a.max_completion_len,
        batch=a.batch * a.num_generations,     # the group is generated in one batch
        grad_accum=a.grad_accum,
    )
    _print_plan(plan)
    print("  ⚠  Note the batch above is multiplied by G. GRPO's memory scales with G —")
    print("     that is the price of dropping the value model. Lower G if you OOM.\n")

    if a.dry_run:
        print("  --dry-run complete.\n")
        return

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if a.quant == "4bit" else None

    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb,
        torch_dtype=torch.bfloat16 if bnb is None else None,
        device_map="auto", attn_implementation="sdpa",
    )
    model.config.use_cache = False

    ds = Dataset.from_list([
        {"prompt": [{"role": "user", "content": r["prompt"]}],
         "answer": r.get("answer", ""),
         "must_contain": r.get("must_contain", [])}
        for r in rows
    ])

    cfg = GRPOConfig(
        output_dir=a.output,
        num_generations=a.num_generations,
        max_prompt_length=a.max_prompt_len,
        max_completion_length=a.max_completion_len,
        temperature=a.temperature,
        top_p=a.top_p,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_iterations=a.num_iterations,
        beta=a.beta,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=a.seed,
        # KL to the reference. TRL computes this against the initial policy.
        loss_type="grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_fn(a.reward)],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(a.output)
    tok.save_pretrained(a.output)

    print(f"\n  ✅ GRPO adapter saved to {a.output}")
    _read_reward_curve(trainer)


def _read_reward_curve(trainer) -> None:
    """
    Interpret the GRPO logs — this is where silent failure shows up.
    """
    hist = trainer.state.log_history
    rewards = [h["reward"] for h in hist if "reward" in h]
    kl = [h.get("kl") for h in hist if h.get("kl") is not None]
    clen = [h.get("completion_length") for h in hist if h.get("completion_length")]

    print("\n  ── reward curve diagnosis ──")
    if len(rewards) >= 4:
        first, last = sum(rewards[:len(rewards) // 4]) / (len(rewards) // 4), \
                      sum(rewards[-len(rewards) // 4:]) / (len(rewards) // 4)
        print(f"  reward  start {first:.3f} → end {last:.3f}  ({last - first:+.3f})")
        if abs(last - first) < 0.01:
            print("  ⚠  Reward is FLAT. Most likely causes, in order:")
            print("       1. pass@G ≈ 0 (task too hard) or ≈ 1 (too easy) → std=0, no advantage")
            print("       2. learning rate too low to move anything")
            print("       3. reward function returning a constant (test it standalone!)")
        elif last < first:
            print("  ⚠  Reward DECREASED. Check for a bug in the reward function, or KL")
            print("     collapse — try raising --beta or lowering --lr.")
        else:
            print("  ✅ Reward is improving.")
    else:
        print("  (not enough logged steps to diagnose)")

    if kl:
        print(f"  KL to reference    final {kl[-1]:.4f}   max {max(kl):.4f}")
        if max(kl) > 20:
            print("  ⚠  KL blew up. The policy has drifted far from the reference — "
                  "this usually precedes degeneration. Raise --beta, lower --lr.")

    if clen:
        growth = clen[-1] / max(clen[0], 1)
        print(f"  completion length  {clen[0]:.0f} → {clen[-1]:.0f} tokens ({growth:.2f}x)")
        if growth > 1.5:
            print("  ⚠  Completions grew >50%. Classic reward-hack signature: the model "
                  "learned that longer answers score higher, not better ones.")

    print("\n  Next: check that generated answers are still READABLE. Reward going up")
    print("  while outputs become repetitive nonsense is the most common GRPO failure.")


if __name__ == "__main__":
    main()
