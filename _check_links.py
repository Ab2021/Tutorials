"""
Link integrity checker for ai-system-design-mastery.

Walks every .md file, extracts every markdown link ](target), resolves
relative targets against the linking file's directory, and reports any
that do not exist on disk.

Skips: http(s)://, mailto:, #anchors, file:

Usage:  python _check_links.py            # whole repo
        python _check_links.py --new      # only the 3 new gap files
"""
import os, re, sys

ROOT = r"D:\AgenticAI\AI_Sys_design\ai-system-design-mastery"

LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# Fenced code blocks contain Python that looks like markdown links to this
# regex -- `TOOLS[name](**args)` and `self.remaining[color]` both match
# `](...)` or `[...]`. Strip fence contents before scanning, or the report
# carries permanent false positives that hide real breakage.
FENCE_RE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", re.M | re.S)


def strip_code(text):
    return FENCE_RE.sub("\n", text)

NEW_FILES = [
    "Case-Studies/HLD/12-netflix-recommendation-and-streaming.md",
    "Case-Studies/HLD/13-fraud-detection-system.md",
    "Case-Studies/HLD/14-autocomplete-and-typeahead.md",
    "HLD/05-ai-systems/10-distributed-model-weight-serving.md",
]


def all_md():
    out = []
    for dp, dn, fs in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in (".git", "node_modules")]
        for f in fs:
            if f.endswith(".md"):
                out.append(
                    os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, "/")
                )
    return sorted(out)


def check(files):
    bad = {}
    total = 0
    for rel in files:
        src = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(src):
            bad.setdefault(rel, set()).add("<<FILE MISSING>>")
            continue
        text = strip_code(open(src, encoding="utf-8").read())
        d = os.path.dirname(rel)
        for m in LINK_RE.finditer(text):
            raw = m.group(1)
            if raw.startswith(("http://", "https://", "mailto:", "#", "file:")):
                continue
            path_part = raw.partition("#")[0]
            if not path_part:
                continue
            total += 1
            target = os.path.normpath(os.path.join(d, path_part)).replace(os.sep, "/")
            if not os.path.exists(os.path.join(ROOT, target.replace("/", os.sep))):
                bad.setdefault(rel, set()).add(raw)
    return total, bad


if __name__ == "__main__":
    files = NEW_FILES if "--new" in sys.argv else all_md()
    total, bad = check(files)
    print("files checked : %d" % len(files))
    print("links checked : %d" % total)
    if bad:
        print("\nBROKEN LINKS: %d file(s)" % len(bad))
        for k in sorted(bad):
            print("  %s" % k)
            for l in sorted(bad[k]):
                print("       -> %s" % l)
    else:
        print("broken links  : NONE")
