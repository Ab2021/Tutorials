"""
common/data_utils.py — dataset loading, formatting, chat templates and loss masking.

This is the module where most SFT bugs live. Read the masking section carefully.

The one idea to take away: **you must mask the prompt tokens out of the loss.**
If you do not, the model is trained to generate the user's question as well as the
assistant's answer, and you get a model that rambles, repeats instructions back, and
scores ~5-15% worse on every benchmark. It is invisible in the loss curve, because
the loss *goes down* — you are just optimizing the wrong objective.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignores this label value.


# --------------------------------------------------------------------------------------
# Format converters — every common SFT data shape, normalised to one internal form.
#
# Internal form: list[{"role": "user"|"assistant"|"system", "content": str}]
# --------------------------------------------------------------------------------------
def from_alpaca(row: dict) -> list[dict]:
    """
    Alpaca / Stanford format — the most common in the wild.

        {"instruction": "...", "input": "...", "output": "..."}

    `input` is optional context ("input" is a terrible name; it is really
    "additional context"). When present it is conventionally appended to the
    instruction. LLaMA-Factory exposes this as columns:
        {"prompt": "instruction", "query": "input", "response": "output"}
    """
    user = row["instruction"]
    if row.get("input"):
        user = f"{user}\n\n{row['input']}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["output"]},
    ]


def from_sharegpt(row: dict) -> list[dict]:
    """
    ShareGPT format. Role names vary between dumps — `from`/`value` is the
    canonical spelling, but `human`/`gpt` and `user`/`assistant` both appear.

        {"conversations": [{"from": "human", "value": "..."},
                           {"from": "gpt",   "value": "..."}]}
    """
    role_map = {
        "human": "user", "user": "user", "system": "system",
        "gpt": "assistant", "assistant": "assistant", "model": "assistant",
    }
    key_role = "from" if "from" in row["conversations"][0] else "role"
    key_val = "value" if "value" in row["conversations"][0] else "content"
    return [
        {"role": role_map.get(t[key_role].lower(), "user"), "content": t[key_val]}
        for t in row["conversations"]
    ]


def from_openai(row: dict) -> list[dict]:
    """
    OpenAI chat format — already in our internal form.

        {"messages": [{"role": "user", "content": "..."}, ...]}
    """
    return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]


def from_completion(row: dict) -> list[dict]:
    """
    Completion-only / continued-pretraining format.

        {"text": "raw document text ..."}

    There is no prompt/answer split here, so *everything* is trained on. That is
    correct for continued pretraining and WRONG for instruction tuning.
    """
    return [{"role": "text", "content": row["text"]}]


FORMAT_LOADERS = {
    "alpaca": from_alpaca,
    "sharegpt": from_sharegpt,
    "openai": from_openai,
    "completion": from_completion,
}


def load_jsonl(path: str | Path, fmt: str = "alpaca", limit: int | None = None) -> list[list[dict]]:
    """Load a .jsonl/.json file and normalise every row to the internal format."""
    loader = FORMAT_LOADERS[fmt]
    rows: list[list[dict]] = []
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(data):
        if limit and i >= limit:
            break
        rows.append(loader(row))
    return rows


# --------------------------------------------------------------------------------------
# Chat templates
# --------------------------------------------------------------------------------------
def render(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """
    Render messages through the tokenizer's chat template.

    NEVER hand-format prompts with f-strings. Every model family has a different
    template (ChatML, Llama-2 [INST], Llama-3 headers, Gemma turns, Mistral), and a
    mismatch between training and inference costs you single-digit-to-double-digit
    benchmark points with no error message. `apply_chat_template` is the only safe
    way to do this.

    Fallback: if the tokenizer has no template (some base models), we emit ChatML,
    which is what most fine-tunes of those models were actually trained on.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# --------------------------------------------------------------------------------------
# Loss masking — the important part
# --------------------------------------------------------------------------------------
def build_masked_example(tokenizer, messages: list[dict], max_len: int = 2048,
                         train_on_prompt: bool = False) -> dict[str, list[int]]:
    """
    Tokenize a conversation and build `labels` that mask out everything except the
    assistant turns.

    Method: render the conversation *incrementally*. After each prefix we know
    exactly how many tokens belong to the prompt so far; the tokens added by the
    assistant's turn are the ones we keep.

    Why incremental and not "tokenize the answer separately"? Because BPE is not
    concatenative — `tokenize(a + b) != tokenize(a) + tokenize(b)` in general. The
    incremental approach sidesteps the off-by-one class of bugs entirely.

    Returns input_ids / attention_mask / labels, truncated to max_len.
    """
    input_ids: list[int] = []
    labels: list[int] = []

    for i, msg in enumerate(messages):
        # Render the conversation up to and including this message, then tokenize.
        prefix = render(tokenizer, messages[: i + 1], add_generation_prompt=False)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]

        if train_on_prompt or msg["role"] == "assistant":
            new_labels = prefix_ids[len(input_ids):]
        else:
            new_labels = [IGNORE_INDEX] * (len(prefix_ids) - len(input_ids))

        input_ids = prefix_ids
        labels.extend(new_labels)

    # Truncate. If we truncate away the assistant turn entirely, the example has no
    # loss signal — caller should filter these out.
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def has_supervision(example: dict) -> bool:
    """True if at least one token contributes to the loss. Use to filter truncated rows."""
    return any(l != IGNORE_INDEX for l in example["labels"])


# --------------------------------------------------------------------------------------
# Preference data (DPO / ORPO / IPO / KTO)
# --------------------------------------------------------------------------------------
def load_preference_jsonl(path: str | Path) -> list[dict]:
    """
    Preference data comes in two shapes:

      * explicit:  {"prompt": ..., "chosen": ..., "rejected": ...}
      * conversational: {"prompt": ..., "chosen": [msgs], "rejected": [msgs]}

    TRL's DPOTrainer accepts both. We normalise to conversational with an explicit
    prompt, which is the least ambiguous.
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row["prompt"]
        prompt_msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        chosen = row["chosen"]
        rejected = row["rejected"]
        out.append({
            "prompt": prompt_msgs,
            "chosen": chosen if isinstance(chosen, list) else [{"role": "assistant", "content": chosen}],
            "rejected": rejected if isinstance(rejected, list) else [{"role": "assistant", "content": rejected}],
        })
    return out


# --------------------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------------------
def pack_examples(examples: list[dict], max_len: int, pad_id: int = 0) -> list[dict]:
    """
    Concatenate short examples into full-length sequences.

    Why: if your average example is 300 tokens and max_len is 2048, padding wastes
    ~85% of every batch. Packing recovers it and typically gives a 2-4x throughput
    win on short-data SFT.

    The trap: naively concatenating lets token i attend to token i-1 *from a
    different document*. That is a real (if mild) quality regression. The correct
    fix is to reset position_ids and use a block-diagonal attention mask; FlashAttention's
    varlen API does this natively. Set `reset_position_ids=True` and pass
    `flash_attn_uses_varlen=True` where your framework supports it.
    """
    packed: list[dict] = []
    buf_ids: list[int] = []
    buf_lbl: list[int] = []

    for ex in examples:
        ids, lbl = ex["input_ids"], ex["labels"]
        if len(buf_ids) + len(ids) > max_len:
            if buf_ids:
                packed.append(_flush(buf_ids, buf_lbl, max_len, pad_id))
                buf_ids, buf_lbl = [], []
            if len(ids) > max_len:  # single example longer than max_len: hard truncate
                ids, lbl = ids[:max_len], lbl[:max_len]
        buf_ids.extend(ids)
        buf_lbl.extend(lbl)

    if buf_ids:
        packed.append(_flush(buf_ids, buf_lbl, max_len, pad_id))
    return packed


def _flush(ids: list[int], lbl: list[int], max_len: int, pad_id: int) -> dict:
    pad = max_len - len(ids)
    return {
        "input_ids": ids + [pad_id] * pad,
        "attention_mask": [1] * len(ids) + [0] * pad,
        "labels": lbl + [IGNORE_INDEX] * pad,
    }


def token_stats(examples: Iterable[dict]) -> dict[str, Any]:
    """Quick sanity report on a dataset — run this before every training job."""
    lens, sup, total_sup = [], 0, 0
    n = 0
    for ex in examples:
        n += 1
        lens.append(len(ex["input_ids"]))
        s = sum(1 for l in ex["labels"] if l != IGNORE_INDEX)
        total_sup += s
        if s:
            sup += 1
    lens.sort()
    return {
        "examples": n,
        "tokens_total": sum(lens),
        "len_p50": lens[n // 2] if n else 0,
        "len_p95": lens[int(n * 0.95)] if n else 0,
        "len_max": lens[-1] if n else 0,
        "with_supervision": sup,
        "without_supervision": n - sup,
        "supervised_token_frac": round(total_sup / max(sum(lens), 1), 3),
    }
