# Course authoring runbook

How the 11-part curriculum is completed, verified, and resumed. Everything here is
filesystem-measured: no step trusts an agent's claim of success.

## What "complete" means

A teaching unit is `<module>/<submodule>` and owns two collections:

- `Case_Studies/` — decision cases, 15 numbered sections each
- `Cheat_Sheets/` — reference cards, 12 numbered sections each, plus one final
  `NN_25_Common_Issues.md`

A unit is **complete** when, for every case and sheet file, `Tools/manuscript_lint.py` exits 0,
and a Common Issues artifact exists with at least 25 `CI-` entries. An index is complete when it
has its required section count (7 for cases, 11 for sheets) and every link resolves.

## Acceptance gate

```bash
python Tools/manuscript_lint.py <file.md>     # exit 0 = accepted
```

It enforces, in order of how often each fires:

| Check | Threshold | Why |
|---|---|---|
| Heading contract | exact numbered H2 set, in order | template conformance |
| Physical length | ≥ 950 lines | depth floor |
| Distinct prose | ≥ 680 non-blank lines | depth floor |
| Long-line repeat | any ≥70-char line at most twice | **the anti-padding rule** |
| Duplicate ratio | < 0.15 | catches mechanical inflation |
| Placeholders | none | no `TODO`/`TBD` |
| Case derivations | ≥3 `FORMULA`/`CONSTRAINT`/`NUMERICAL EXAMPLE` blocks | checked arithmetic |
| Common Issues | ≥25 `CI-` entries | real troubleshooting |

The repeat and duplicate-ratio checks are what reject the previous pass's 1,015-line files that
repeated one sentence hundreds of times. Length alone was never the acceptance criterion.

## Pipeline

```text
Tools/build_queue.py      scan disk -> queue.jsonl + per-task briefs (887 tasks)
Tools/plan_batches.py     one manuscript per batch, no file owned twice (887 batches)
Tools/dispatch_next.py    claim N batches, print self-contained prompts
Tools/run_batch.py        shared prompt text + --verify (re-lints everything)
Tools/gen_indexes.py      deterministic 7/11-section collection indexes
Tools/sync_unit_indexes.py deterministic unit topic-map status
Tools/verify_course.py    link integrity across all Markdown
Tools/coverage_report.py  per-part accepted/manuscript counts
```

`gen_indexes` and `sync_unit_indexes` require **no model**: they read frontmatter and H1 titles from
the real files, so an index can never link a file that does not exist.

## Resume after an interruption

State lives in `.authoring_state/` (protected; see contract section 10).

```bash
python Tools/build_queue.py      # recompute gaps from disk
python Tools/plan_batches.py     # rebuild conflict-free batches
python Tools/run_batch.py --verify
python Tools/dispatch_next.py --slots 12      # next 12 prompts
python Tools/coverage_report.py               # where we stand
```

Because the queue is derived from the filesystem, a resumed session re-does nothing that already
passes lint and re-queues everything that does not. Deleting a state file is therefore recoverable;
deleting manuscripts is not.

## Ownership rules that prevent corruption

- One batch owns one manuscript file. Batches never share a path (verified at planning time).
- Workers may not touch root indexes, `00_*` files, `Templates/`, `Tools/`, or any other subtree.
- Indexes are generated centrally, after manuscripts land, from what actually exists.
- Workers build a manuscript in memory and write it in one call, so an interrupted run cannot leave
  a truncated file.

## Known evidence discipline

- Constructed numbers are labelled `ILLUSTRATIVE` and must be re-derived, not quoted.
- No invented company architecture or measured outcome.
- A company name in a catalog is a research context, not a claim about its systems.
- Unavailable sources stay unavailable and are recorded as gaps beside the affected conclusion.
