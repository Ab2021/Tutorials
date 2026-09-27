#!/usr/bin/env python
"""
07_distillation.py — compress a large teacher into a small student.

The three things distillation actually gives you
-----------------------------------------------
Fine-tuning a small model on hard labels teaches it WHAT to output. Distillation teaches
it HOW the teacher distributes probability mass, and that extra signal is the whole point:

  1. **Dark knowledge.** A teacher's softmax over "cat / dog / lynx" carries the
     information that a lynx is a near-miss. A hard label "cat" throws that away. Hinton's
     original result: on MNIST, distilling a teacher into a student recovered most of the
     accuracy of an *ensemble* while being a single model.
  2. **A regulariser.** Soft targets are a smoother objective than one-hot labels, so the
     student overfits less on small datasets.
  3. **Capacity transfer.** A 0.5B student can learn behaviour from a 70B teacher that it
     could never have learned from 5k hard-labelled examples, because the teacher supplies
     a dense signal on every token of every sequence.

The temperature parameter, and the trap in it
---------------------------------------------
    p_i = softmax(z_i / T)

T = 1 is the model's own distribution. T > 1 flattens it, exposing the relative ordering
of the non-argmax classes — that is where the dark knowledge lives. Typical T is 2-5.

The trap: the gradient magnitude of the KD loss scales as 1/T^2, so if you raise T and
keep the loss weight fixed, the distillation term quietly stops mattering. Hinton's fix is
to multiply the KD term by T^2 to restore the gradient scale:

    loss = alpha * T^2 * KL(teacher_T || student_T)  +  (1 - alpha) * CE(student, hard)

**alpha weights the SOFT (KD) term.** That is Hinton's convention and the one CH-08 §4
documents. `--alpha 0.7` therefore means 70% distillation / 30% hard labels, which is a
sensible default (Hinton used 0.9 for the soft term on MNIST). Getting this backwards is
a silent failure: the run trains, the loss falls, and the student has quietly learned
mostly from the hard labels — i.e. you have paid for a teacher and run plain SFT.

(The first version of this script had it backwards, which meant `--alpha 0.7` — the exact
command CH-08 §6 tells you to run — produced 70% hard / 30% KD. The mixture is now printed
explicitly on every run so the convention is never in doubt again.)

This script prints both terms separately, so you can always see which one is carrying the
gradient.

Which distillation to use
-------------------------
  * **Sequence-level KD** (Kim & Rush 2016) — generate the teacher's output, treat it as a
    hard label, run ordinary SFT. No logits needed, works with any API model, and is what
    most "distillation" in practice means today. Weaker signal, vastly cheaper.
  * **Token-level / response KD** — the classic soft-target loss above. Needs the teacher's
    full logit distribution, so teacher and student must share a tokenizer (or you must
    align vocabularies, which is a project in itself).
  * **Feature / hidden-state KD** — match intermediate activations, usually with a learned
    projection when the hidden sizes differ (FitNets). Strong, but requires architectural
    access to both models.
  * **Relation KD (RKD)** — match the *relationships* between examples rather than the
    outputs. Robust to teacher-student capacity gap.

Run it
------
    python 07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \\
        --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl --dry-run
    python 07_distillation.py --from-teacher --teacher ... --out data/seqkd.jsonl
    python 07_distillation.py --token-kd --teacher <big> --student <small> --text data/corpus.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  — UTF-8 console fix
from common.memory import inference_gb, kv_cache_gb       # noqa: E402

DEFAULTS = {
    "temperature": 2.0,
    "alpha": 0.7,          # weight on the SOFT (KD) term; (1-alpha) goes to hard CE.
                           # Hinton's convention, and CH-08 §4's. 0.7 = 70% KD / 30% CE.
    "top_k": 100,          # teacher logits to keep — see the storage note below
    "max_new_tokens": 512,
}


# --------------------------------------------------------------------------------------
# Mode 1: sequence-level KD — the practical default
# --------------------------------------------------------------------------------------
def run_from_teacher(a) -> None:
    """
    Generate teacher responses and write them as an SFT dataset.

    This is the one to reach for first. It needs no logits, no shared tokenizer, and no
    access to teacher weights at all — the teacher can be a hosted API. The price is that
    the student only sees the teacher's argmax, not its uncertainty.
    """
    prompts = _load_prompts(a.prompts, a.n, a.dry_run)
    print(f"\n  ── sequence-level KD ──")
    print(f"  teacher            {a.teacher}")
    print(f"  prompts            {len(prompts)}")
    # framework_overhead_gb=0 because this line claims to be the WEIGHTS. The default
    # 1.0 GB of framework overhead and the 1-token KV cache belong to the other two
    # lines; folding them in made a 29.8 GiB teacher print as "30.8 GB weights".
    print(f"  teacher weights    ~{inference_gb(a.size_hint, 1, 1, a.quant_bits, framework_overhead_gb=0):.1f} GB "
          f"(at --quant-bits {a.quant_bits})")
    print(f"  teacher KV cache   ~{kv_cache_gb(a.size_hint, a.max_new_tokens, 1, a.quant_bits):.2f} GB "
          f"(for {a.max_new_tokens} tokens, batch of 1)")
    print(f"  max_new_tokens     {a.max_new_tokens}")

    if a.dry_run:
        est_out = len(prompts) * a.max_new_tokens
        print(f"\n  est. output tokens ~{est_out:,}")
        print(f"  est. teacher time  ~{est_out / 2000 / 60:.0f} min at 2k tok/s "
              "(batch of 1 — batched vLLM is 10-30x faster)")
        print("\n  --dry-run: nothing generated.")
        print("\n  ⚠  BEFORE YOU GENERATE 5,000 RESPONSES:")
        print("     Generate 20 first and READ them. Teacher generations that are")
        print("     repetitive, over-long, or refuse the prompt will be faithfully")
        print("     imitated. Distillation copies the teacher's flaws as reliably as")
        print("     its strengths — including its verbosity and its hedging.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(a.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        a.teacher, torch_dtype="auto", device_map="auto")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, t0 = 0, time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=a.max_new_tokens,
                                     do_sample=True, temperature=a.teacher_temp,
                                     top_p=0.95, pad_token_id=tok.pad_token_id)
            resp = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            if len(resp.split()) < 5:
                continue
            f.write(json.dumps({"instruction": p, "input": "", "output": resp},
                               ensure_ascii=False) + "\n")
            kept += 1
            if (i + 1) % 50 == 0:
                el = time.perf_counter() - t0
                print(f"    {i+1}/{len(prompts)}  kept {kept}  "
                      f"{el:.0f}s elapsed  eta {el/(i+1)*(len(prompts)-i-1):.0f}s")

    print(f"\n  ✅ wrote {kept} teacher responses to {out_path}")
    print(f"     ({len(prompts) - kept} responses were dropped as too short to be useful)")
    print("\n  ⚠  These are RAW teacher outputs. Refusals, repeated boilerplate and")
    print("     instruction-echoing all survive generation and all teach the student bad")
    print("     habits. There is no --filter flag here: read a sample yourself.")
    print("     `data/make_instruction_data.py` has a quality filter, but it only runs")
    print("     on data that script generates in the same process — it cannot filter an")
    print("     existing jsonl, so pointing it at this file will not work.")
    print("     To filter these outputs, read 20 and write the rule you actually need.")
    print(f"\n     python 01_sft_lora.py --data {out_path} --output out/student")


# --------------------------------------------------------------------------------------
# Mode 2: token-level KD — the real thing, and its constraints
# --------------------------------------------------------------------------------------
def run_token_kd(a) -> None:
    # Transformers is needed even for --dry-run: the vocabulary check below is THE
    # go/no-go for token-level KD, and it needs the real tokenizers. Torch and the
    # weights are NOT needed for the plan, so they load only on the training path.
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        sys.exit(f"  pip install transformers  ({e})")

    print(f"\n  ── token-level KD ──")
    tok_t = AutoTokenizer.from_pretrained(a.teacher)
    tok_s = AutoTokenizer.from_pretrained(a.student)

    if _vocab(tok_t) != _vocab(tok_s):
        sys.exit(
            f"  Teacher vocab ({_vocab(tok_t):,}) != student vocab ({_vocab(tok_s):,}).\n\n"
            "  Token-level KD needs a shared vocabulary: the loss compares two probability\n"
            "  distributions defined over the SAME index space. Options:\n"
            "    1. Pick a teacher/student pair from the same family (Qwen2.5-32B → Qwen2.5-0.5B).\n"
            "    2. Use sequence-level KD (--from-teacher) — it has no such constraint.\n"
            "    3. Align the vocabularies explicitly (real work; usually not worth it)."
        )

    print(f"  teacher            {a.teacher}")
    print(f"  student            {a.student}")
    print(f"  shared vocab       {_vocab(tok_t):,} tokens ✓")
    print(f"  T                  {a.T}   (KD gradient scales as 1/T^2 — we multiply by T^2 below)")
    print(f"  alpha              {a.alpha} on the SOFT (KD) term / "
          f"{1 - a.alpha:.1f} on hard CE")

    text = Path(a.text).read_text(encoding="utf-8", errors="ignore")
    # A plain list, not a tensor: the plan only needs the token COUNT, and keeping
    # torch out of this path lets --dry-run run with no GPU and no torch installed.
    ids = tok_t(text)["input_ids"]
    print(f"  corpus             {len(ids):,} tokens")

    print("\n  ── the storage problem nobody mentions ──")
    print("    Caching the teacher's FULL logits for a 150k vocab costs, per token:")
    print(f"      150,000 x 4 bytes (fp32) = 600 KB/token")
    print(f"    For {len(ids):,} tokens that is "
          f"{len(ids)*600_000/1e12:.2f} TB. You cannot cache this.")
    print(f"    Top-{a.top_k} truncation: {a.top_k} x (4 bytes logit + 4 bytes index) = "
          f"{a.top_k*8/1024:.1f} KB/token → {len(ids)*a.top_k*8/1e9:.2f} GB. Cached.")
    print("    This is why real KD pipelines store top-k logits, not full distributions,")
    print("    and why the teacher runs OFFLINE, once, not alongside training.")

    if a.dry_run:
        # Everything above is checkable WITHOUT the weights: vocab alignment (the real
        # go/no-go for token-level KD), the corpus size, and the storage plan. Loading a
        # 32B teacher just to print a plan would defeat the point of --dry-run.
        print("\n  --dry-run: stopped before loading the models.")
        print(f"     would load   teacher {a.teacher}")
        print(f"                  student {a.student}")
        try:
            hint = a.size_hint.strip().upper()
            n = float(hint.rstrip("BM"))
            n_teacher = n * (1e9 if hint.endswith("B") else 1e6)
            print(f"     teacher size ~{n_teacher * 2 / 1e9:.1f} GB of fp16 weights "
                  f"(from --size-hint {a.size_hint})")
            print("                   + KV cache, + the student, + activations")
            print("     Tip: --quant-bits 0.5 (int4) cuts that ~4x if you only need")
            print("          the logits, which is all KD ever uses the teacher for.")
        except ValueError:
            print(f"     teacher size ~ (could not parse --size-hint {a.size_hint!r})")
        print("\n  ℹ  The vocab check above is the ONE thing to settle before planning")
        print("     anything else. If it passed, token-level KD is viable: cache the")
        print("     teacher's top-k logits offline ONCE, then train the student against")
        print("     those cached tensors rather than keeping both models resident.")
        return

    # ── past this point we need torch and the real weights ──────────────────────────
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    t_model = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype="auto",
                                                   device_map="auto").eval()
    s_model = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype="auto",
                                                   device_map="auto").train()

    # DEVICE — do NOT assume both models landed on the same card. `device_map="auto"`
    # places each model (and, for a sharded model, each layer) wherever it found room,
    # and `cuda:0` is a guess that is simply wrong when the teacher filled GPU 0 and the
    # student was pushed to GPU 1. Take the real device from each model's first parameter.
    t_dev = next(t_model.parameters()).device
    s_dev = next(s_model.parameters()).device
    if t_dev != s_dev:
        print(f"  ℹ  teacher on {t_dev}, student on {s_dev} — moving the top-k support "
              f"across, which is cheap because it is only {a.top_k} logits per position.")

    seq = min(a.seq_len, len(ids))
    if seq < 2:
        sys.exit(f"  The corpus tokenises to {len(ids)} tokens; at least 2 are needed to "
                 f"form a next-token target.")
    ids = torch.tensor(ids, dtype=torch.long)

    with torch.no_grad():
        t_logits = t_model(ids[:seq].unsqueeze(0).to(t_dev)).logits[0]
    s_logits = s_model(ids[:seq].unsqueeze(0).to(s_dev)).logits[0]

    # Top-k truncation, then renormalise. Note the ordering: we select with the teacher,
    # then gather the SAME indices from the student, so both distributions live on the
    # same support. torch.gather requires index and input on one device, so the teacher's
    # indices AND values both come across to the student's device.
    top = torch.topk(t_logits, a.top_k, dim=-1)
    t_idx = top.indices.to(s_dev)
    t_top = top.values.to(s_dev)
    s_top = torch.gather(s_logits, -1, t_idx)

    T = a.T
    # Both terms must average over the SAME positions, or alpha does not mean what it says.
    # The hard term is next-token CE over tokens 1..seq-1. The KD term must be the same
    # range: position 0 has no hard target, and position seq-1 predicts a token we do not
    # have. Slicing both to 1..seq-1 makes the two means comparable.
    t_top = t_top[1:]
    s_top = s_top[1:]

    t_soft = F.softmax(t_top / T, dim=-1)
    s_log = F.log_softmax(s_top / T, dim=-1)

    # T^2 because d(softmax(z/T))/dz scales as 1/T. Without it, raising T silently
    # shrinks the KD term's gradient and the student drifts back to plain SFT.
    # reduction="sum" then divide by the SAME n the CE mean divides by — using
    # reduction="batchmean" here would divide by `top_k`, not by the position count.
    n_pos = t_top.shape[0]
    kd = F.kl_div(s_log.reshape(-1, a.top_k), t_soft.reshape(-1, a.top_k),
                  reduction="sum") / n_pos * (T * T)

    # `ids` is CPU-resident now (it may have to feed a model on any device), so the
    # targets come across to the student's device explicitly.
    hard = F.cross_entropy(
        s_logits[1:].reshape(-1, s_logits.size(-1)).float(),
        ids[:seq][1:].reshape(-1).to(s_dev),
    )
    loss = a.alpha * kd + (1 - a.alpha) * hard

    print(f"\n  ── loss terms at init (one forward pass, no training) ──")
    print(f"    hard CE          {hard.item():.4f}")
    print(f"    KD (x T^2)       {kd.item():.4f}")
    print(f"    mixture          {a.alpha:.2f} x KD + {1-a.alpha:.2f} x CE "
          f"= {loss.item():.4f}   ← --alpha weights the SOFT term")
    print(f"    positions        {n_pos} (both terms average over the same {n_pos} tokens)")
    if a.alpha > 0 and kd.item() < hard.item() * 0.1:
        print("    ℹ  KD is <10% of CE even though it carries "
              f"{a.alpha:.0%} of the weight.")
        print("       Either T is too low to expose dark knowledge, or the teacher and")
        print("       student are already very close. Check that the top-k indices are")
        print("       aligned before changing anything else.")
    print("\n  ℹ  Sanity check: if the teacher and student were the SAME model, KD would be")
    print("     0. So KD > 0 here is expected. But a KD near 0 with a DIFFERENT teacher")
    print("     means the truncation or the index alignment is wrong.")
    print("\n  Wire this loss into your training loop. Caching t_idx/t_top to disk once and")
    print("  reusing them across student runs is the standard way to amortise teacher cost.")
    print("  Storage: top-k fp32 logit + int32 index = 8 bytes per entry, so "
          f"{a.top_k:,} x 8 B = {a.top_k*8/1024:.1f} KB/token.")


# --------------------------------------------------------------------------------------
def _vocab(tok) -> int:
    return len(tok)


def _load_prompts(path: str | None, n: int, dry_run: bool) -> list[str]:
    if path:
        p = Path(path)
        if not p.exists():
            sys.exit(f"  --prompts not found: {p}")
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        out = []
        for r in rows:
            pr = r.get("instruction") or r.get("prompt") or r.get("question") or ""
            if r.get("input"):
                pr = f"{pr}\n\n{r['input']}"
            if pr.strip():
                out.append(pr.strip())
        if not out:
            sys.exit(f"  {p} has {len(rows)} rows but none yielded a prompt. Expected one "
                     f"of 'instruction', 'prompt' or 'question'.")
        return out[:n]

    if not dry_run:
        # Hard stop. The previous version fell through to ten placeholder strings and,
        # without --dry-run, generated and WROTE a 10-row "dataset" of sentences reading
        # "Placeholder prompt 3 — replace with --prompts." A warning was printed and
        # execution continued, so the failure mode was a silently useless training set.
        sys.exit(
            "  --prompts is required for --from-teacher.\n"
            "  Expected a jsonl where each row has 'instruction', 'prompt' or 'question'\n"
            "  (optionally with 'input'). Generate one with:\n"
            "      python data/make_instruction_data.py --out data/prompts.jsonl\n"
            "  Or pass --dry-run to see the plan without generating anything.")
    print("  ℹ  No --prompts file: using a 10-prompt placeholder set for --dry-run only.")
    return [f"Placeholder prompt {i} — replace with --prompts." for i in range(10)]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-teacher", action="store_true",
                      help="Sequence-level KD: generate teacher outputs, SFT the student")
    mode.add_argument("--token-kd", action="store_true",
                      help="Token-level KD: compare softmax distributions (needs shared vocab)")

    p.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompts", default=None, help="jsonl of prompts (--from-teacher)")
    p.add_argument("--text", default=None, help="raw .txt corpus (--token-kd)")
    p.add_argument("--out", default="data/seqkd.jsonl")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("-T", "--temperature", dest="T", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--alpha", type=float, default=DEFAULTS["alpha"],
                   help="Weight on the SOFT (KD) term; (1-alpha) goes to the hard-label CE. "
                        "Hinton's convention, and the one CH-08 §4 uses. 0.7 means 70%% "
                        "distillation / 30%% ground truth.")
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    p.add_argument("--teacher-temp", type=float, default=0.8,
                   help="SAMPLING temperature for generation (≠ the KD T)")
    p.add_argument("--size-hint", default="32B", help="Teacher size label for the VRAM estimate")
    p.add_argument("--quant-bits", type=float, default=1.0, help="2=fp16, 1=int8, 0.5=int4")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.token_kd and not a.text:
        sys.exit("--token-kd requires --text <corpus.txt>")
    if a.T <= 0:
        sys.exit("T must be > 0 (T=1 means no softening)")

    if a.from_teacher:
        run_from_teacher(a)
    else:
        run_token_kd(a)


if __name__ == "__main__":
    main()
