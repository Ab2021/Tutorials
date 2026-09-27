#!/usr/bin/env python
"""
14_vertex_gemini_finetune.py — supervised fine-tuning on Google Vertex AI.

Vertex's schema is NOT OpenAI's
-------------------------------
This is the first thing that bites people, and the error messages are unhelpful.

  OpenAI                        Vertex AI
  ----------------------------  ------------------------------------------
  {"messages": [...]}           {"messages": [...]}   (same key, different rules)
  role: "system"                role: "system"  (allowed as the FIRST message only)
  {"role","content"}            {"role","content"}
  file purpose "fine-tune"      a JSONL object in GCS, referenced by URI
  model id string               a tunedModel resource name

The differences that actually cost you a run:
  * Vertex wants the file in **Google Cloud Storage**, not uploaded via API. You need a
    bucket and the right IAM role before you can even start.
  * Vertex's `role` values are validated strictly; an unexpected role fails the whole job.
  * The tuned model is a **long-lived resource** that you must explicitly DEPLOY to an
    endpoint before you can call it — and, critically, **UNDEPLOY** when you are done.
    An idle deployed endpoint bills by the hour. This is the #1 surprise on the invoice.
  * Adapters have a size; the tuning mode you pick determines both cost and how much the
    model can change.

The lifecycle, in order
-----------------------
    dataset (GCS)  →  tuning job  →  tuned model  →  deploy to endpoint  →  predict
                                                  →  UNDEPLOY  ← do not skip

Run it
------
    python 14_vertex_gemini_finetune.py --data data/sft.jsonl --validate
    python 14_vertex_gemini_finetune.py --data data/sft.jsonl --estimate
    python 14_vertex_gemini_finetune.py --data data/sft.jsonl --upload \\
        --project my-proj --bucket my-bucket
    python 14_vertex_gemini_finetune.py --train --project my-proj --bucket my-bucket
    python 14_vertex_gemini_finetune.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  — UTF-8 console fix

# Per-1M-token USD. Verify before budgeting — Google changes these, and the tuned-endpoint
# serving premium is billed per HOUR of deployment, not per token, which changes the maths
# completely from OpenAI's model.
PRICES = {
    "gemini-1.5-flash": {"train": 3.00, "input": 0.075, "output": 0.30,
                         "endpoint_hour": 3.00},
    "gemini-1.5-pro":   {"train": 8.00, "input": 1.25,  "output": 5.00,
                         "endpoint_hour": 5.00},
    "gemini-2.0-flash": {"train": 3.00, "input": 0.10,  "output": 0.40,
                         "endpoint_hour": 3.00},
}

DEFAULT_MODEL = "gemini-1.5-flash-002"
HOURS_PER_MONTH = 730


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="jsonl in Vertex chat format")
    p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    p.add_argument("--location", default="us-central1")
    p.add_argument("--bucket", default=os.environ.get("VERTEX_BUCKET"))
    p.add_argument("--model", default=DEFAULT_MODEL, help="Base model id")
    p.add_argument("--display-name", default="handbook-sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr-multiplier", type=float, default=1.0)
    p.add_argument("--adapter-size", type=int, default=4, choices=[1, 4, 8, 16, 32],
                   help="LoRA rank for the tuning adapter; higher = more capacity, more cost")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--list", action="store_true", help="List tuned models and endpoints")
    p.add_argument("--undeploy", default=None, metavar="ENDPOINT_ID",
                   help="UNDEPLOY an endpoint so it stops billing")
    p.add_argument("--inference-volume", type=int, default=100_000)
    p.add_argument("--avg-output-tokens", type=int, default=250)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.list:
        return _list(a)
    if a.undeploy:
        return _undeploy(a)
    if not a.data and (a.validate or a.estimate or a.upload):
        sys.exit("--validate/--estimate/--upload require --data")

    rows = _load(a.data) if a.data else []

    if a.validate or not any([a.estimate, a.upload, a.train]):
        _validate(rows, a)
    if a.estimate or not any([a.upload, a.train]):
        _estimate(rows, a)
    if a.upload:
        _upload(rows, a)
    if a.train:
        _train(a)


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"  No such file: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------------------
def _validate(rows: list[dict], a) -> None:
    print(f"\n  ── validating {len(rows)} examples (Vertex schema) ──")
    problems: dict[str, int] = {}

    def flag(k: str) -> None:
        problems[k] = problems.get(k, 0) + 1

    tot_in = tot_out = 0
    for i, r in enumerate(rows):
        msgs = _to_vertex(r)
        if msgs is None:
            flag("not_convertible_to_vertex_shape")
            continue
        if not msgs:
            flag("empty_messages")
            continue

        roles = [m["role"] for m in msgs]
        if msgs[0]["role"] == "system" and "system" in roles[1:]:
            # Vertex allows a system instruction, but only in the leading position.
            flag("system_message_not_first")
        if "model" not in roles and "assistant" not in roles:
            flag("no_model_turn")
        elif roles[-1] not in ("model", "assistant"):
            flag("last_turn_is_not_the_model")
        for m in msgs:
            if m["role"] not in ("system", "user", "model", "assistant"):
                flag(f"bad_role:{m['role']}")
            if not str(m.get("content", "")).strip():
                flag("empty_content")

        tot_in += sum(len(m["content"].split()) for m in msgs[:-1])
        tot_out += len(msgs[-1]["content"].split())

    if len(rows) < 10:
        flag("too_few_examples_for_a_tuning_job")

    est_in, est_out = int(tot_in * 1.33), int(tot_out * 1.33)

    print(f"  examples           {len(rows)}")
    print(f"  est. tokens        {est_in + est_out:,}  (in {est_in:,} / out {est_out:,})")

    if problems:
        print(f"\n  ⚠  issues: {dict(sorted(problems.items(), key=lambda kv: -kv[1]))}")
        print("     Vertex fails the ENTIRE job on a schema violation rather than skipping")
        print("     the offending row, so a single bad line costs you a full run.")
    else:
        print("\n  ✅ schema looks valid")

    print("\n  ── the GCS prerequisite ──")
    if a.bucket:
        print(f"  bucket             gs://{a.bucket}  ✓")
    else:
        print("  ⚠  No --bucket. Vertex cannot read a local file. Create one and grant")
        print("     the Vertex service account read access:")
        print("       gsutil mb -l us-central1 gs://YOUR_BUCKET")
        print("       gcloud projects add-iam-policy-binding PROJECT \\")
        print("         --member=serviceAccount:service-PROJECT_NUMBER@")
        print("         gcp-sa-aiplatform.iam.gserviceaccount.com \\")
        print("         --role=roles/storage.objectViewer")
    print("\n  ℹ  Vertex wants the model turn under role 'model' (not 'assistant') in some")
    print("     SDK versions. This script normalises it on upload; confirm the resulting")
    print("     file if the job rejects yours.")


def _to_vertex(r: dict) -> list[dict] | None:
    """Normalise any reasonable on-disk shape to Vertex's messages list."""
    msgs = None
    if "messages" in r and isinstance(r["messages"], list):
        msgs = r["messages"]
    elif "instruction" in r and "output" in r:
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        ins = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        msgs += [{"role": "user", "content": ins},
                 {"role": "model", "content": r["output"]}]
    elif "prompt" in r and "completion" in r:
        msgs = [{"role": "user", "content": r["prompt"]},
                {"role": "model", "content": r["completion"]}]
    if msgs is None:
        return None
    # Normalise 'assistant' → 'model' for the tuned-model training format.
    return [{"role": ("model" if m.get("role") == "assistant" else m.get("role")),
             "content": str(m.get("content", ""))} for m in msgs]


# --------------------------------------------------------------------------------------
def _estimate(rows: list[dict], a) -> None:
    key = next((k for k in PRICES if k in a.model), None)
    if key is None:
        print(f"\n  ⚠  No price on file for '{a.model}'. Known: {list(PRICES)}")
        return
    pr = PRICES[key]

    tot_in = tot_out = 0
    for r in rows:
        msgs = _to_vertex(r)
        if not msgs:
            continue
        tot_in += sum(len(m["content"].split()) for m in msgs[:-1])
        tot_out += len(msgs[-1]["content"].split())
    est_in, est_out = int(tot_in * 1.33), int(tot_out * 1.33)

    print(f"\n  ── cost estimate ({key}) ──")
    train_tokens = (est_in + est_out) * a.epochs
    train_cost = train_tokens / 1e6 * pr["train"]
    print(f"\n  TUNING (one-off)")
    print(f"    tokens           {train_tokens:,}  ({a.epochs} epochs)")
    print(f"    cost             ${train_cost:,.2f}")

    # THE decisive difference from OpenAI: a deployed tuned endpoint bills per HOUR,
    # whether or not anyone calls it.
    hourly = pr["endpoint_hour"]
    monthly_fixed = hourly * HOURS_PER_MONTH
    print(f"\n  SERVING (recurring, and it bills while IDLE)")
    print(f"    endpoint         ${hourly:.2f}/hour x {HOURS_PER_MONTH} h = "
          f"${monthly_fixed:,.2f}/month")
    print(f"    ↑ This is charged whether or not you send a single request.")

    calls = a.inference_volume
    inf_in = int(calls * est_in / max(len(rows), 1))
    inf_out = int(calls * a.avg_output_tokens)
    per_token = inf_in / 1e6 * pr["input"] + inf_out / 1e6 * pr["output"]
    print(f"\n  per-token usage  at {calls:,} calls/month")
    print(f"    ${per_token:,.2f}/month")

    total_m = monthly_fixed + per_token
    print(f"\n  TOTALS")
    print(f"    month 1          ${train_cost + total_m:,.2f}")
    print(f"    year 1           ${train_cost + total_m*12:,.2f}")
    print(f"    fixed endpoint   {monthly_fixed/total_m*100:.0f}% of the recurring bill")

    print("\n  ── the decision this arithmetic forces ──")
    # Compare the actual dollars, not a call-count threshold. The crossover depends on
    # the model's prices and your average prompt/output sizes, so a hard-coded call count
    # would be wrong for every configuration but one.
    if monthly_fixed > per_token:
        ratio = monthly_fixed / max(per_token, 0.01)
        print(f"    The FIXED endpoint charge is {ratio:.0f}x the usage charge at")
        print(f"    {calls:,} calls/month (${monthly_fixed:,.0f} vs ${per_token:,.2f}).")
        print("    You are paying for an idle endpoint. Either UNDEPLOY between batches")
        print("    and redeploy when you need it, or batch requests into sessions rather")
        print("    than leaving it running around the clock.")
        crossover = int(calls * ratio) if per_token > 0 else None
        if crossover:
            print(f"    Usage would need to reach ~{crossover:,} calls/month before the")
            print("    per-token term caught up with the idle charge.")
    else:
        print(f"    At {calls:,} calls/month the per-token term "
              f"(${per_token:,.2f}) exceeds the fixed endpoint charge "
              f"(${monthly_fixed:,.2f}).")
        print("    Volume is high enough that a self-hosted open model — which has no")
        print("    idle-hour charge — is worth costing out before you commit.")
    print("\n    ALWAYS:  --undeploy ENDPOINT_ID when you are done. This is the single")
    print("    most common way a Vertex fine-tuning experiment produces a surprise bill.")
    print("    Model weights themselves are cheap to STORE; it is the DEPLOYMENT that costs.")


# --------------------------------------------------------------------------------------
def _require_gcp(a) -> None:
    missing = [n for n, v in [("--project", a.project), ("--bucket", a.bucket)] if not v]
    if missing:
        sys.exit(f"  Missing {', '.join(missing)}. Set GOOGLE_CLOUD_PROJECT and "
                 f"VERTEX_BUCKET, or pass them explicitly.")


def _upload(rows: list[dict], a) -> None:
    _require_gcp(a)
    from google.cloud import storage

    local = Path(a.data).with_suffix(".vertex.jsonl")
    with local.open("w", encoding="utf-8") as f:
        n = 0
        for r in rows:
            msgs = _to_vertex(r)
            if not msgs:
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    print(f"\n  normalised {n} rows → {local}")

    uri = f"gs://{a.bucket}/vertex/{local.name}"
    client = storage.Client(project=a.project)
    client.bucket(a.bucket).blob(f"vertex/{local.name}").upload_from_filename(str(local))
    print(f"  ✅ uploaded to {uri}")
    print(f"\n     python 14_vertex_gemini_finetune.py --train --project {a.project} "
          f"--bucket {a.bucket}")


def _train(a) -> None:
    _require_gcp(a)
    try:
        import vertexai
        from vertexai.tuning import sft
    except ImportError:
        sys.exit("  pip install google-cloud-aiplatform")

    vertexai.init(project=a.project, location=a.location)
    uri = f"gs://{a.bucket}/vertex/{Path(a.data or 'data').stem}.vertex.jsonl"

    print(f"\n  starting tuning job")
    print(f"    base model       {a.model}")
    print(f"    training data    {uri}")
    print(f"    epochs           {a.epochs}")
    print(f"    lr multiplier    {a.lr_multiplier}")
    print(f"    adapter size     {a.adapter_size}")

    job = sft.train(
        source_model=a.model,
        train_dataset=uri,
        tuned_model_display_name=a.display_name,
        epochs=a.epochs,
        learning_rate_multiplier=a.lr_multiplier,
        adapter_size=a.adapter_size,
    )
    print(f"  ✅ job started: {job.resource_name}")
    print("\n  ── adapter size: the knob people ignore ──")
    print("    It sets the LoRA rank of the tuning adapter, i.e. HOW MUCH the model is")
    print("    allowed to change. 1 is cheapest and most constrained; 32 is the most")
    print("    capacity. For style/format adaptation 4-8 is usually plenty; large values")
    print("    on small datasets overfit. It also drives the size — and cost — of the")
    print("    artifact you deploy.")
    print("\n  ── next steps ──")
    print(f"    python 14_vertex_gemini_finetune.py --list")
    print("    Then deploy to an endpoint to call it:")
    print("      from vertexai.preview import tuning")
    print("      tuned.deploy()")
    print("\n  ⚠  AND UNDEPLOY WHEN DONE:")
    print("      python 14_vertex_gemini_finetune.py --undeploy ENDPOINT_ID")


def _list(a) -> None:
    try:
        import vertexai
        from google.cloud import aiplatform
    except ImportError:
        sys.exit("  pip install google-cloud-aiplatform")
    if not a.project:
        sys.exit("  --project required")
    vertexai.init(project=a.project, location=a.location)
    aiplatform.init(project=a.project, location=a.location)

    print("\n  ── tuned models ──")
    for m in aiplatform.Model.list():
        print(f"    {m.display_name:<30} {m.resource_name}")
    print("\n  ── endpoints (THESE COST MONEY WHILE DEPLOYED) ──")
    eps = list(aiplatform.Endpoint.list())
    if not eps:
        print("    none — good, nothing is billing")
    for e in eps:
        print(f"    {e.display_name:<30} {e.resource_name}")
    print("\n  Any endpoint listed above is billing per hour right now, whether or not")
    print("  it is receiving traffic. Undeploy the ones you are not actively using.")


def _undeploy(a) -> None:
    try:
        from google.cloud import aiplatform
    except ImportError:
        sys.exit("  pip install google-cloud-aiplatform")
    if not a.project:
        sys.exit("  --project required")
    aiplatform.init(project=a.project, location=a.location)
    ep = aiplatform.Endpoint(a.undeploy)
    print(f"  undeploying {ep.display_name}...")
    ep.undeploy_all()
    print("  ✅ undeployed — the hourly endpoint charge has stopped")
    print("     The tuned model itself is still stored and can be redeployed later.")


if __name__ == "__main__":
    main()
