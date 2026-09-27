#!/usr/bin/env python3
"""Knowledge-base verifier.

Runs the plan's checks 1, 2 and 4 across whatever exists so far, so the build stays
resumable and each wave boundary is verified the same way.

    python verify.py            # summary
    python verify.py -v         # per-file detail

Checks
  1. Structural  — every Tnn ID present in each family; files non-trivial; blueprints complete
  2. Provenance  — every file has a `Transcript coverage:` line and a `## Sources` block;
                   every `refs/...` path named in any file resolves on disk
  4. Links       — every relative markdown link resolves

Check 3 (code runs) and 5 (no fabrication) are not automatable here: run each
`03-design-blueprints/*/run.py` by hand, and spot-check corpus numbers against the transcripts.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

TOPICS = [
    "T01-sampling-decoding", "T02-search-decoding", "T03-constrained-generation",
    "T04-test-time-compute", "T05-verifiers-best-of-n", "T06-inference-fundamentals",
    "T07-kv-cache", "T08-batching-scheduling", "T09-speculative-decoding",
    "T10-quantization", "T11-parallelism-moe", "T12-disaggregation-kv-transfer",
    "T13-serving-engines", "T14-routing-gateways", "T15-autoscaling-slo",
    "T16-agentic-inference", "T17-observability-evals", "T18-guardrails-security",
    "T19-finops-sovereignty",
]
IDS = [t.split("-")[0] for t in TOPICS]

FAMILIES = {
    "cheat sheet": ("00-cheat-sheets", "{slug}.md", 400),
    "case study": ("01-case-studies", "{slug}.md", 400),
    "interview": ("02-interview-questions", "{slug}.md", 400),
}
MIN_WORDS = 400
VERBOSE = "-v" in sys.argv

problems = []
notes = []


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ---------------------------------------------------------------- 1. structural
def check_structural():
    for family, (d, pat, minw) in FAMILIES.items():
        if not os.path.isdir(d):
            problems.append(f"[1 struct] missing family directory: {d}")
            continue
        have = 0
        for slug in TOPICS:
            f = os.path.join(d, pat.format(slug=slug))
            if not os.path.exists(f):
                notes.append(f"[1 struct] not yet written: {rel(f)} ({family})")
                continue
            have += 1
            w = len(open(f, encoding="utf-8").read().split())
            if w < minw:
                problems.append(f"[1 struct] thin ({w}w < {minw}): {rel(f)}")
        if VERBOSE:
            print(f"  {family:12s} {have}/{len(TOPICS)}")

    # blueprints need the full set of artifacts to count
    for slug in TOPICS:
        d = os.path.join("03-design-blueprints", slug)
        if not os.path.isdir(d):
            problems.append(f"[1 struct] missing blueprint dir: {d}")
            continue
        need = ["HLD.md", "LLD.md", "run.py", "docs/SEQUENCES.md"]
        for n in need:
            if not os.path.exists(os.path.join(d, n)):
                notes.append(f"[1 struct] blueprint pending: 03-design-blueprints/{slug}/{n}")
        if os.path.isdir(os.path.join(d, "production")) and not os.listdir(os.path.join(d, "production")):
            notes.append(f"[1 struct] blueprint pending: 03-design-blueprints/{slug}/production/ (empty)")


# ---------------------------------------------------------------- 2. provenance
def check_provenance():
    n_paths = 0
    n_files = 0
    for d in ("00-cheat-sheets", "01-case-studies", "02-interview-questions", "03-design-blueprints"):
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "**", "*.md"), recursive=True):
            if os.path.basename(f).upper().startswith("README"):
                continue
            t = open(f, encoding="utf-8").read()
            n_files += 1
            if "Transcript coverage:" not in t:
                problems.append(f"[2 prov] no `Transcript coverage:` line: {rel(f)}")
            if "## Sources" not in t:
                problems.append(f"[2 prov] no `## Sources` block: {rel(f)}")
            for p in re.findall(r"`(refs/[^`\s]+)`", t):
                n_paths += 1
                if not os.path.exists(p):
                    problems.append(f"[2 prov] unresolvable source path in {rel(f)}:\n           {p}")
    if VERBOSE:
        print(f"  {n_files} files, {n_paths} refs/ paths checked")


# ---------------------------------------------------------------- 4. links
def planned_paths():
    """Every path this build is *supposed* to produce. A link pointing at one of these is
    pending, not broken — which keeps the verifier usable while waves are in flight."""
    out = set()
    for _, (d, pat, _) in FAMILIES.items():
        for slug in TOPICS:
            out.add(os.path.normpath(os.path.join(d, pat.format(slug=slug))))
    for slug in TOPICS:
        d = os.path.join("03-design-blueprints", slug)
        for n in ("HLD.md", "LLD.md", "run.py", "docs/SEQUENCES.md"):
            out.add(os.path.normpath(os.path.join(d, n)))
    out.add("README.md")
    out.add("TOPICS.md")
    out.add("TEMPLATES.md")
    out.add("PROGRESS.md")
    return out


def check_links():
    planned = planned_paths()
    n = 0
    for d in ("00-cheat-sheets", "01-case-studies", "02-interview-questions", "03-design-blueprints"):
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "**", "*.md"), recursive=True):
            t = open(f, encoding="utf-8").read()
            for target in re.findall(r"\]\((?!https?:|#)([^)]+)\)", t):
                target = target.split("#")[0].strip()
                if not target:
                    continue
                n += 1
                resolved = os.path.normpath(os.path.join(os.path.dirname(f), target))
                if os.path.exists(resolved):
                    continue
                if resolved in planned:
                    notes.append(f"[4 link] pending target: {rel(f)} -> {target}")
                else:
                    problems.append(f"[4 link] broken link in {rel(f)}: {target}")
    if VERBOSE:
        print(f"  {n} relative links checked")


if __name__ == "__main__":
    print("=" * 72)
    print("Knowledge-base verification")
    print("=" * 72)
    check_structural()
    check_provenance()
    check_links()

    if problems:
        print(f"\nPROBLEMS ({len(problems)})\n")
        for p in problems:
            print("  " + p)
    else:
        print("\nNo problems found in checks 1, 2 and 4.")

    if notes:
        print(f"\nPending work ({len(notes)} items) — expected while waves are in flight:")
        shown = notes if VERBOSE else notes[:12]
        for p in shown:
            print("  " + p)
        if not VERBOSE and len(notes) > 12:
            print(f"  ... and {len(notes) - 12} more (run with -v)")

    print("\nNot automatable here:")
    print("  [3 code]  run each 03-design-blueprints/*/run.py manually")
    print("  [5 fabr]  spot-check corpus numbers against refs/ transcripts")
    sys.exit(1 if problems else 0)
