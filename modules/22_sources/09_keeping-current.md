# Keeping Current

> Module 22 — Sources and Further Reading
> Topic 09 of 09
> Estimated study time: 2.5 hours
> Prerequisites: 01_how-to-read-this-list
> Version 2026.3

## 1. Why this topic exists

Every other topic in this module teaches how to judge a source at a moment. This topic teaches how to keep that judgement true. Sources decay: guidance is revised, standards are amended, code sets are released, datasets are versioned, papers are retracted, and the sentence that was correct at drafting becomes wrong at submission. The decay is silent, and the cost lands on whoever reuses the sentence next.

The failures are ordinary:

- A regulatory summary is updated every year, and a guidance is revised two months after the update.
- A slide deck cites a standard edition that was withdrawn, because the deck was built from a template.
- A reading list points to a paper that was retracted, and every new analyst inherits the citation.
- A model card describes the dataset version used in development, and the production pipeline moved to a new release.
- A team monitors the FDA but not the MDCG, and the EU launch plan is built on a superseded FAQ.
- A quarterly review checks links but not status, so a page that still resolves is counted as current.

The people affected are the regulatory lead who signs a stale summary, the analyst who reuses an old number, the new hire who trusts the reading list, and the organisation that discovers the problem during an audit rather than before it.

Maintenance is also the part of source work most often skipped, because it has no visible output. A review that finds nothing looks like a waste of an afternoon, and the value only appears in the failure that did not happen.

This topic turns maintenance into a routine with a watch list, a cadence, a triage rule, a quarterly checklist, and a retirement process. It ends the module because it is the practice that keeps the other eight topics from expiring.

> **Note:** Currency is a property of a source at a date, not a property of a document. A summary can be recent and wrong, and an old document can be current. The only reliable check is against the issuer, at the review date.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** a watch list with sources, owners, cadences, and triggers. Assessed by Exercise 1 and self-check item 1.
2. **Assign** a review cadence from the volatility of the source class rather than from habit. Assessed by Exercise 2.
3. **Apply** a triage rule that separates changes requiring action from changes to watch or ignore. Assessed by Exercise 3 and section 7.
4. **Run** a quarterly review that checks status, not only links. Assessed by Exercise 4 and the worked example.
5. **Propagate** a change to every document and model that depends on the source. Assessed by Exercise 5.
6. **Retire** outdated material without losing the record of what it said. Assessed by Exercise 6.
7. **Estimate** the effort of a maintenance routine and fit it to a realistic monthly budget. Assessed by Exercise 7 and section 7.
8. **Design** a twelve-month maintenance calendar for a team. Assessed by Exercise 8.

## 3. The core mental model

Maintenance is a loop with four stages: watch, triage, act, and retire. The loop runs on a cadence set by the source's volatility, and every action updates a register that the next cycle reads.

```text
                    WATCH LIST
        source | class | owner | cadence | trigger
                              |
                              v
                    DETECT CHANGE
        new release | revision | retraction | deprecation
                              |
                              v
                    TRIAGE
        act now | schedule | watch | ignore
                              |
                              v
                    ACT AND PROPAGATE
        update documents, models, registers
                              |
                              v
                    RETIRE AND ARCHIVE
        supersede, date-stamp, keep the record
                              |
                              v
                    QUARTERLY AUDIT
        check status at the issuer, not the link
```

Read the loop as a cycle, not a project. The output of the audit is the input to the next watch cycle, and the register is the memory that makes the cycle cheap.

**Cadence follows volatility.** A statute changes rarely; guidance changes yearly; a code set changes annually on a fixed date; a dataset can change without notice. A single annual review is either too often for the statute or too rare for the dataset.

**Detection is not the same as reading.** A feed or an alert tells you that something changed. It does not tell you whether the change affects you. The triage step is where most of the effort should go, because reading every update in full is unsustainable.

**Propagation is the hard part.** A change to a guidance can affect a regulatory summary, a model card, a validation plan, a sales deck, and a training module. If the register does not link sources to documents, the change is applied in one place and missed in four.

**Retirement is a source-management act.** An outdated document should be superseded, date-stamped, and archived rather than deleted, because the record of what was believed at a date is itself useful and because an audit may ask for it.

| Source class | Typical cadence | Trigger examples | Effort per cycle |
|---|---|---|---|
| Regulation | Annual, or on legislative event | Amendment, transition deadline | 2 hours |
| Guidance | Quarterly | New draft, revision, withdrawal | 1 hour per source |
| Standard | Semiannual | New edition, amendment | 1 hour |
| Code set | On release date | Annual or periodic release | 2 hours |
| Dataset | On use, and on release | New version, licence change | 1 hour |
| Literature | Monthly alert review | Retraction, contradicting trial | 2 hours |
| Market | Quarterly | New report, revised figures | 1 hour |

## 4. Definitions and vocabulary

### 4.1 Maintenance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Watch list | The set of sources monitored with owners and cadences. | A folder of bookmarks | Register |
| Cadence | The interval between reviews. | Uniform across all sources | Months |
| Trigger | An event that forces an earlier review. | Replaced by a calendar date | Event |
| Triage | Sorting changes by the action they require. | Reading everything in full | Decision |
| Propagation | Updating every dependent document. | Updating the summary only | Action |
| Retirement | Superseding a source and recording why. | Deleting the old file | Status |
| Archive | A dated copy kept for the record. | A live URL | Snapshot |
| Source register | The record of sources, versions, and uses. | A bibliography | Database |
| Drift | Divergence between a summary and its source. | Treated as a writing issue | Status |
| Change log | The issuer's record of revisions. | Assumed to exist | Document |

### 4.2 Monitoring terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Alert | A notification of a new item. | Treated as an assessment | Notification |
| Feed | A machine-readable list of updates. | Assumed comprehensive | RSS or API |
| Changelog | A dated list of changes. | Ignored in favour of the latest page | Document |
| Release note | A description of a specific release. | Confused with a changelog | Document |
| Deprecation | A notice that a source will be retired. | Treated as immediate removal | Status |
| Sunset date | The date a source stops being supported. | Confused with a retraction | Date |
| Superseded | Replaced by a newer version. | Treated as withdrawn | Status |
| Withdrawn | Removed by the issuer. | Treated as superseded | Status |
| Retracted | Withdrawn for validity reasons. | Confused with a correction | Status |
| Expression of concern | A notice raising a question. | Ignored until retraction | Status |

### 4.3 Archive and record terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Snapshot | A dated copy of a page or document. | A screenshot without a date | Archive copy |
| Permalink | A stable link to a specific version. | A live homepage URL | URL |
| DOI | A persistent identifier. | Assumed immutable content | Identifier |
| Version pin | Recording the exact version used. | Recording "latest" | Version string |
| Retention period | How long a record must be kept. | Set by convenience | Years |
| Audit trail | The record of who changed what and when. | Absent from shared drives | Log |
| Tombstone | A record that a source existed and was retired. | Deleted file | Record |
| Re-verification | Checking the source again at the issuer. | Checking the link only | Action |

### 4.4 Review and governance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Quarterly review | A scheduled check of the watch list. | A link check | Process |
| Owner | The person accountable for a source. | A shared inbox | Role |
| Escalation | Raising a material change to a decision-maker. | Applied to every change | Process |
| Materiality | Whether a change affects a decision or document. | Assumed for every change | Judgement |
| Register review | A periodic audit of the register itself. | Assumed unnecessary | Process |
| Coverage | The proportion of sources with an owner and cadence. | Counted as the number of sources | Proportion |

### 4.5 Signals that maintenance is failing

| Signal | What it suggests | What to check |
|---|---|---|
| Same sources reviewed every quarter | The list is stale | Coverage and ownership |
| No source ever changes status | The check is a link check | Status at the issuer |
| Documents cite different versions | Propagation is missing | Source-to-document links |
| New hires use old templates | Retirement is missing | Template review |
| Review takes days | Cadence is too coarse | Triage and automation |
| Review takes minutes | Coverage is too thin | Sample the register |

## 5. The mechanics, step by step

The procedure has 20 steps in five phases. It builds the loop in section 3 and leaves a routine that fits a monthly budget.

The routine is designed for a team of one to ten people. A larger organisation can distribute the steps by role, and a smaller one can run them in a single monthly hour.

### Phase A — Build the watch list

**Step 1. List the sources the work depends on.**
Input: active documents, models, submissions, and training material.
Output: every external source cited, with the document that cites it.
Rationale: the watch list should be derived from use, not from interest.
Branch: if a source is cited in no document, it is optional and can be dropped.
Edge case: a source can be cited indirectly through a template, which is still a use.
Check: is every citation mapped to a document.
Record: the source-to-document map.

**Step 2. Classify each source by type.**
Input: the list.
Output: regulation, guidance, standard, code set, dataset, literature, market, community.
Rationale: the type determines the cadence and the trigger.
Branch: a source that mixes types, such as a guidance with a code annex, should be split.
Edge case: a page can host several documents with different statuses.
Check: is every source classified.
Record: the classification.

**Step 3. Assign an owner to each source.**
Input: the team roles.
Output: one named person per source or source group.
Rationale: an unowned source is not monitored, and a shared inbox is not an owner.
Branch: if no one can own a group, reduce the list rather than leave it unowned.
Edge case: an owner can be external, such as a regulatory consultant, with a contract.
Check: does every source have a name.
Record: the owners.

**Step 4. Set the cadence and the triggers.**
Input: the type and the decision horizon.
Output: an interval and a list of events.
Rationale: cadence follows volatility, and triggers catch the changes that matter before the interval expires.
Branch: a source with a known release date should be reviewed on that date, not on a quarterly cycle.
Edge case: an election or a court decision can change a timeline without a release.
Check: does every source have both.
Record: the cadence and triggers.

### Phase B — Detect changes

**Step 5. Choose the detection method for each source.**
Input: the issuer's channels.
Output: a feed, an alert, a mailing list, or a manual check.
Rationale: the method should match how the issuer communicates, not how the team prefers to read.
Branch: if the issuer has no feed, use a saved search or a calendar reminder.
Edge case: some issuers announce changes only in an official journal.
Check: is the method recorded.
Record: the method.

**Step 6. Subscribe and test the method.**
Input: the chosen channels.
Output: a confirmed subscription and a test notification.
Rationale: an unverified subscription is a hope, not a method.
Branch: if a feed is noisy, filter by keyword rather than abandoning it.
Edge case: an alert can silently stop when a page is restructured.
Check: has a test notification arrived.
Record: the subscription.

**Step 7. Collect changes into one queue.**
Input: the notifications.
Output: a dated list of changes with links.
Rationale: a single queue prevents a change from being handled twice or not at all.
Branch: if the queue grows faster than it is processed, tighten the filters.
Edge case: duplicate notifications are common and should be de-duplicated by URL.
Check: is there one queue.
Record: the queue.

**Step 8. Check the issuer's changelog for unannounced changes.**
Input: the issuer's changelog or revision history.
Output: changes that were not announced.
Rationale: not every revision is announced, and the changelog is the authoritative list.
Branch: if there is no changelog, compare the current page with the archived snapshot.
Edge case: a changelog can be incomplete or undated.
Check: is the changelog reviewed.
Record: the unannounced changes.

### Phase C — Triage

**Step 9. Classify each change by impact.**
Input: the change and the documents that cite the source.
Output: high, medium, or low impact.
Rationale: impact is a property of the dependency, not of the change's prominence.
Branch: a change to a definition used in a label is high impact even if the document is short.
Edge case: a change can be high impact for one team and none for another.
Check: is the impact tied to a document.
Record: the impact.

**Step 10. Apply the triage rule.**
Input: the impact and the source type.
Output: act now, schedule, watch, or ignore.
Rationale: the rule prevents both over-reaction and neglect.
Branch: high impact is always act now; low impact on a non-regulatory source is ignore.
Edge case: a medium-impact code change can become high if the code is used in a cohort definition.
Check: is the action one of the four.
Record: the action.

**Step 11. Estimate the effort for each action.**
Input: the change and the affected documents.
Output: hours per action.
Rationale: the estimate makes the load visible and prevents an unsustainable promise.
Branch: if the total exceeds the budget, defer the low-impact items explicitly.
Edge case: an action can require a meeting, which is more than the reading time.
Check: is the effort recorded.
Record: the hours.

**Step 12. Decide what to read in full.**
Input: the action list.
Output: the items to read completely and the items to skim.
Rationale: reading everything is not sustainable, and reading nothing is not maintenance.
Branch: act-now items are read in full; watch items are skimmed for relevance.
Edge case: a summary can be enough for a watch item and never enough for an act-now item.
Check: is the reading depth recorded.
Record: the reading plan.

### Phase D — Act and propagate

**Step 13. Update the source register.**
Input: the change and the new version.
Output: an updated register entry with the date and the change.
Rationale: the register is the record that the loop depends on.
Branch: if the source is superseded, record the successor and the reason.
Edge case: a source can change without a version number, which is why the date is recorded.
Check: is the register current.
Record: the entry.

**Step 14. Propagate to every dependent document.**
Input: the source-to-document map.
Output: an updated list of documents and the changes made.
Rationale: a change applied in one place and missed in four is worse than no update, because it creates inconsistent documents.
Branch: if a document cannot be updated immediately, mark it as stale and date the mark.
Edge case: a template can carry the stale text into future documents.
Check: are all dependents updated or marked.
Record: the propagation list.

**Step 15. Propagate to models and pipelines.**
Input: the model register and the pipeline code.
Output: a statement of whether the change affects features, labels, or thresholds.
Rationale: a definition or code change can silently alter a model's inputs.
Branch: if the change affects a label, re-run the validation.
Edge case: a code release can change a cohort without changing the rule.
Check: is the model impact assessed.
Record: the model impact.

**Step 16. Escalate material changes.**
Input: the act-now items.
Output: a decision-maker informed with the change, the impact, and the recommendation.
Rationale: material changes often require a decision that the source owner cannot make alone.
Branch: a change to a regulatory timeline is material to a launch plan.
Edge case: an escalation can be a single paragraph, not a meeting.
Check: is the escalation recorded.
Record: the escalation.

### Phase E — Retire and audit

**Step 17. Retire superseded sources.**
Input: the register.
Output: a status change, a successor link, and a tombstone entry.
Rationale: retirement keeps the register honest and prevents reuse of the old source.
Branch: if the source is needed for a historical claim, keep it with a date and a "historical" label.
Edge case: a source can be withdrawn without a successor, which leaves a gap to record.
Check: is every superseded source marked.
Record: the retirement.

**Step 18. Archive what was read.**
Input: the page or document.
Output: a dated snapshot and a permalink.
Rationale: the archive is the evidence of what the source said at the review date.
Branch: if the source is paywalled, archive the citation and the passage, not the whole document.
Edge case: an archive can be the only remaining copy after a withdrawal.
Check: is the snapshot dated.
Record: the archive.

**Step 19. Run the quarterly audit.**
Input: the register and a sample of sources.
Output: a coverage statement, a status check, and a list of gaps.
Rationale: the audit checks the register itself, not only the sources.
Branch: if coverage is below the target, reduce the list rather than the checks.
Edge case: an audit should sample sources that have not changed, because they are the ones most likely to be unchecked.
Check: is the audit documented.
Record: the audit.

**Step 20. Review the routine's cost and adjust.**
Input: the hours spent and the changes found.
Output: an adjusted cadence or list.
Rationale: a routine that costs more than the risk it manages will be abandoned.
Branch: if the routine is too expensive, automate detection and keep human triage.
Edge case: a quiet quarter is not evidence that the routine is unnecessary.
Check: is the cost recorded.
Record: the cost.

The procedure produces a routine that is cheap enough to sustain and complete enough to catch the changes that matter. It does not eliminate the risk of a missed change, and it makes the risk visible and bounded.

### The five tests applied at every review

Maintenance is not only detection. At each review, every source on the list is re-tested with the same five questions that were used to admit it, because a source can pass on the day it is added and fail a year later.

| Test | Question at review | What failure looks like | Response |
|---|---|---|---|
| Authority | Is the issuer still the body that owns this subject | A guidance has been taken over by another body | Re-attribute or replace |
| Currency | Is this the version in force, and is a successor listed | A revision is dated after the last check | Re-verify and propagate |
| Applicability | Does it still bear on a decision the team makes | The programme it supported has closed | Downgrade or retire |
| Accessibility | Can a colleague still obtain it | A subscription has lapsed | Archive the passage or replace |
| Provenance | Is the claim traceable to a primary document | The trail ends at a summary | Chase the primary source |

The tests are ordered by cost, not by importance. Authority and currency are quick to check; provenance is slow and is reserved for sources that carry a claim into a document.

**Branch: the source passes all five.** Record the date and leave it in place. A passing source still needs its cadence honoured, because the next review is the one that will catch the change.

**Branch: the source fails authority.** If the issuing body has changed, the source is not simply stale; it may be wrong about who governs the subject. Replace it and check whether the successor has a different scope.

**Branch: the source fails currency.** Establish whether a successor exists and whether it applies to work already done. The response is one of three: update and propagate, document the earlier basis, or pause the dependent work until the question is settled.

**Branch: the source fails applicability.** A source that no longer bears on a decision is a candidate for retirement, and retirement is a legitimate outcome. Keeping it costs review time and dilutes attention.

**Branch: the source fails accessibility.** If the team can no longer reach the document, the correct action is to archive what was relied on and find an accessible equivalent, not to keep citing a page nobody can open.

**Branch: the source fails provenance.** A summary that cannot be traced is not usable as evidence, even if it is convenient. Either find the primary document or downgrade the claim it supports.

### Edge cases at review

| Edge case | Signal | Correct action | Common error |
|---|---|---|---|
| Retracted paper | Retraction notice or a status flag at the publisher | Remove the claim, replace the citation, note the date | Keeping the citation because the PDF is still online |
| Superseded guidance | A successor is listed by the issuer | Re-read the successor and record the change | Citing the old version because it is familiar |
| Paywalled standard | Access ends when a subscription lapses | Archive the clause relied on, or buy access | Citing a clause nobody on the team can check |
| Dead link | The page returns an error or redirects | Find the new location or archive the last good copy | Leaving the link because it worked last year |
| Predatory journal | Structural signals in the venue | Remove and replace, and check other citations from it | Keeping it because a colleague cited it |
| Preprint versus peer-reviewed | A journal version now exists | Cite the published version and check for changed results | Citing the preprint after publication |
| Version behind a DOI | The DOI resolves to a newer version | Record the version used and the date | Citing the DOI as if it pinned the content |
| Withdrawn dataset | The maintainer announces retirement | Re-extract from the successor or document the freeze | Continuing to quote counts from a withdrawn release |

### A decision table for the review

| Finding | Evidence | Action | Owner |
|---|---|---|---|
| In force, no successor | Issuer's status page | Keep, record date | Source owner |
| Successor published | Issuer's index | Re-verify, propagate | Source owner and artefact owner |
| Withdrawn, no successor | Issuer's notice | Retire, document basis | Source owner |
| Unreachable, archived copy available | Snapshot plus citation | Keep with archive reference | Source owner |
| Unreachable, no copy | Failed retrieval twice | Retire, record gap | Register owner |
| Disputed status | Two readers disagree | Escalate to regulatory owner | Regulatory owner |

Three outcomes are not enough for this decision, because "keep" and "retire" leave out the common case in which the source is still valid but the team cannot reach it. The table therefore separates availability from validity, and the archive column carries the distinction.

### The quarterly checklist

- Confirm that every source on the list still exists at the issuer.
- Check status, not only the link.
- Check the changelog for unannounced revisions.
- Review the alert queue and clear it.
- Confirm that every act-now item was propagated.
- Confirm that every superseded source is marked.
- Sample five documents for stale citations.
- Sample five models for version drift.
- Record the hours spent.
- Record what changed and what did not.
- Update the cadence if a source proved quieter or noisier than expected.
- Report coverage to the owner.

The checklist takes about two hours for a list of thirty sources, which is the budget the routine is designed for.

### Archive checklist

- Capture the page or document as it appeared.
- Record the date and the reviewer.
- Record the version or document number.
- Store the snapshot where the team can find it.
- Link the snapshot from the register.
- Note any passage relied on.
- Re-check that the archive opens.

An archive that nobody can open is not an archive, and a snapshot without a date is not evidence.

### What the routine costs

| Activity | Frequency | Hours per cycle | Hours per year |
|---|---|---:|---:|
| Alert triage | Weekly | 0.5 | 26 |
| Quarterly checklist | Quarterly | 2 | 8 |
| Regulatory deep check | Semiannual | 4 | 8 |
| Register maintenance | Monthly | 0.5 | 6 |
| Archive and retirement | Quarterly | 1 | 4 |
| Annual review of the routine | Annual | 3 | 3 |

The total is about 55 hours a year, or roughly one hour a week. That is the realistic cost of keeping a thirty-source register current, and it is small relative to the cost of one stale regulatory claim.

The estimate excludes the first build of the register, which takes about two days for a mature programme and is the largest single cost of the routine.

### Detection methods compared

| Method | Latency | Completeness | Effort | Best for |
|---|---|---|---|---|
| Official feed | Minutes to days | High for that issuer | Low | Regulators with feeds |
| Mailing list | Days | Medium | Low | Guidance updates |
| Saved search | Days | Medium | Low | Literature |
| Changelog review | On demand | High | Medium | Standards and code sets |
| Manual page check | On demand | Depends on the page | High | Issuers without feeds |
| Community mention | Variable | Low | Low | Early signal, not verification |

A feed tells you that a page changed. It does not tell you whether the change matters, and it does not cover issuers without feeds.

### Triage examples

| Change | Impact | Action | Why |
|---|---|---|---|
| Guidance revision affecting a claim | High | Act now | The claim may be unsupported |
| Code release adding a new code | Medium to high | Schedule | The cohort may change |
| New edition of a textbook | Low to medium | Watch | Only therapy chapters matter |
| Benchmark paper with a new result | Low | Watch | No decision depends on it |
| Retraction in the reading list | High | Act now | A document may cite it |
| Journal policy change | Low | Ignore | No local dependency |

The rule is not a substitute for judgement. It is a way to make the judgement consistent across reviewers and to keep the volume manageable.

### A propagation map

| Source change | Document | Model or pipeline | Action |
|---|---|---|---|
| Definition revision | Data dictionary, model card | Label logic | Re-derive labels |
| Threshold change | Validation plan, slide deck | Decision threshold | Re-run evaluation |
| Code release | Cohort specification | Extraction query | Re-extract and compare |
| Guidance revision | Regulatory summary, submission | None | Update and re-review |
| Dataset release | Data register, methods section | Training pipeline | Re-train or re-report |
| Retraction | Any document citing it | None | Replace and annotate |

The map is the artefact that turns a source change into a work list. Without it, propagation depends on someone remembering where the source was used.

### A branch worked in miniature: the quiet quarter

A quarterly review finds no changes. The team concludes that the routine is unnecessary and skips the next quarter.

The next quarter brings two changes: a guidance revision that affects a claim, and a code release that changes a cohort. Neither is caught, and both are discovered in a review.

The branch is not about the quiet quarter; it is about what the quiet quarter means. A quiet quarter is evidence that the issuers were quiet, not that the routine is redundant. The correct response is to record the quiet quarter and keep the cadence, or to lengthen it with a trigger that fires on a release.

The alternative branch is to automate detection and keep human triage. That reduces the cost of a quiet quarter without removing the coverage.

- A quiet quarter is not a reason to stop.
- A busy quarter is not a reason to lengthen the cadence.
- The cadence follows the source, not the workload.

A quiet quarter should be recorded as a finding, because it is the only evidence that the cadence is proportionate.

Recording it also protects the routine from being cut in a budget review, when the absence of incidents is read as the absence of risk.

The same logic applies to a busy quarter. A spike in changes is evidence about the field, not about the routine, and the response is to triage more carefully rather than to review more often. Adding cadence under pressure produces a burst of reading that is not sustained, and the following quarter returns to the previous interval with the backlog unread.

- Record both quiet and busy quarters with the same fields.
- Change the cadence only when the source cadence changes.

## 6. Worked example

A regulatory summary on the intranet carries the line: "Last updated 12 September 2026. Current." The team relies on it for a launch decision in November.

### Framing the claim

The sentence "current" is a claim about status, and it is supported by a date. The date is real; the status is not established by the date.

| Sub-claim | Type | What could support it |
|---|---|---|
| The page was edited on 12 September | Fact | The content management system |
| The content reflects the issuer's position | Status | A check at the issuer |
| The position is still in force | Currency | The issuer's current list |
| The summary is adequate for the decision | Judgement | A check of the specific provision |

### Candidate sources

| # | Candidate | Type | Role |
|---|---|---|---|
| C1 | The intranet summary | Secondary | The object under test |
| C2 | The issuer's guidance page | Primary | Current status |
| C3 | The Federal Register notice | Primary | Issue date and availability |
| C4 | An archived snapshot from September | Archive | What the summary read |
| C5 | A colleague's recollection | Anecdote | The origin of "nothing changed" |
| C6 | A vendor newsletter | Secondary | The likely source of a claim that it changed |

### Scoring the candidates

**C1, the summary.**
- Authority: internal, secondary.
- Currency: the edit date is recent; the content may not be.
- Applicability: written for the team's decision.
- Accessibility: internal.
- Provenance: the author and their sources are not recorded.
- Verdict: usable only with a status check.

**C2, the issuer page.**
- Authority: highest for status.
- Currency: current at the access date.
- Applicability: general.
- Accessibility: public.
- Provenance: official.
- Verdict: primary support for status.

**C3, the Federal Register.**
- Authority: high for the issue date and the notice.
- Currency: current.
- Applicability: general.
- Accessibility: public.
- Provenance: official.
- Verdict: primary support for the date.

**C4, the archive.**
- Authority: high for what the page said.
- Currency: historical.
- Applicability: the summary's content.
- Accessibility: the archive service.
- Provenance: dated snapshot.
- Verdict: evidence of the summary's basis.

**C5, the recollection.**
- Authority: low.
- Currency: unknown.
- Applicability: personal.
- Accessibility: a conversation.
- Provenance: none.
- Verdict: not evidence.

**C6, the newsletter.**
- Authority: low to moderate.
- Currency: current.
- Applicability: general.
- Accessibility: public.
- Provenance: secondary.
- Verdict: a prompt to check C2.

### The check

The check is two minutes of work and produces a different answer from the date:

- The issuer's guidance page shows a revision dated 2 October 2026.
- The revision changes the transition provision that the summary describes.
- The summary was edited on 12 September and did not include the revision.
- The archive confirms that the summary matched the pre-revision page.

The summary was current when it was written. It is not current at the review date, and the edit date is not evidence otherwise.

### Verdict

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| The page was edited in September | Fact | "The summary was last edited on 12 September 2026." |
| The content is current | Reject | "The issuer revised the guidance on 2 October 2026; the summary predates the revision." |
| Adequate for the November decision | Reject until updated | Requires the revised transition provision |

The corrected line is: "Last checked against the issuer on 26 October 2026; revision of 2 October incorporated." That line is longer and it is the only one that supports a decision.

### Two diagnostics that would change the verdict

**Diagnostic 1: the revision does not affect the cited provision.** If the October revision changed an unrelated annex, the summary's relevant content could still be current, and the verdict would shift to "current for the provision cited, with the revision noted". The diagnostic is a comparison of the revision's scope with the provision relied on.

**Diagnostic 2: the summary's author performed a status check.** If the author recorded a check at the issuer on 12 September, and the issuer's revision came later, the summary was current on its date and the failure is the absence of a trigger, not the author's work. The verdict becomes a process finding: the routine lacked a trigger for the issuer's revision.

A third diagnostic, weaker, is a public announcement of the revision before 12 September. It would show that the summary was stale when written, which is a different failure with a different fix.

### What the team should record

The register records the summary, its edit date, the issuer page, the revision date, and the access date. The summary is updated with the revision and a check line. The trigger is added: any revision to the issuer's guidance forces a review of the summary within five working days.

The case produces a rule: "last updated" is a fact about a file, not a status. A status line must name the issuer and the date of the check.

### Status lines compared

| Status line | What it asserts | What it establishes |
|---|---|---|
| "Updated September 2026" | The file was edited | Nothing about the issuer |
| "Reviewed internally" | Someone read it | Nothing about status |
| "Current" | A claim | Nothing without a source |
| "Checked against the issuer, 26 October 2026" | A source and a date | Status at that date |
| "Version 2.4, issuer revision of 2 October" | A version and a revision | Status and change |

Only the last two lines are auditable. The first three are common and none of them can be checked.

### What a status check requires

- The issuer's current list, not a search result.
- The document number and the revision date.
- The specific provision relied on.
- The access date and the reviewer.
- A note of any revision that affects the provision.

A status check that does not name the provision is a check of the document, not of the claim.

### Triggers that force an early check

| Trigger | Why it matters | Where to watch |
|---|---|---|
| Issuer revision | The summary may be stale | The issuer's guidance list |
| New draft guidance | The position may change | The guidance database |
| Transition deadline | The applicable route changes | The regulation's transitional articles |
| Court or legislative action | The instrument may be vacated | The official journal |
| Code release | The cohort may change | The code set release notes |
| Retraction | A citation may be invalid | The retraction database |

Each trigger should be linked to the sources it affects, so that a notification becomes a work item rather than a headline.

### What the reviewer records

- The source and the provision checked.
- The issuer's page and the access date.
- The revision date and the version.
- The verdict: current, revised, superseded, or unknown.
- The documents affected.
- The next check date.

The record is short, and it is the only thing that distinguishes a status check from a read.

The difference between a check and a read is the output. A read produces an impression; a check produces a dated line that another person can audit.

A check is also cheaper than a read, because it looks at one provision rather than a whole document. That is the reason the routine can cover thirty sources in two hours.

## 7. Code, query, or calculation

The script turns the watch list into a quarterly work plan. It classifies each change with the triage rule, sums the effort by quarter, and prints the load so that the routine can be sized before it is promised.

```python
"""Maintenance triage for a source watch list."""

from dataclasses import dataclass

HOURS = {"high": 4.0, "medium": 1.5, "low": 0.5}


@dataclass
class Change:
    source: str
    kind: str          # guidance | standard | code | dataset | paper
    impact: str        # high | medium | low
    quarter: int


def triage(c):
    if c.impact == "high":
        return "act now"
    if c.impact == "medium" and c.kind in ("guidance", "standard", "code"):
        return "schedule"
    if c.impact == "medium":
        return "watch"
    return "ignore"


ITEMS = [
    Change("FDA guidance revision", "guidance", "high", 1),
    Change("MDCG guidance update", "guidance", "high", 1),
    Change("ICD-10-CM annual release", "code", "high", 4),
    Change("LOINC release", "code", "medium", 2),
    Change("SNOMED release", "code", "medium", 2),
    Change("HL7 FHIR ballot", "standard", "medium", 3),
    Change("CDISC version", "standard", "low", 3),
    Change("ICH E9(R1) Q&A", "guidance", "medium", 1),
    Change("Dataset version 3.2", "dataset", "high", 2),
    Change("Retraction in reading list", "paper", "high", 3),
    Change("New benchmark paper", "paper", "low", 3),
    Change("Society guideline", "guidance", "high", 4),
    Change("EMA guideline", "guidance", "medium", 2),
    Change("ISO standard amendment", "standard", "medium", 4),
    Change("Journal changes policy", "paper", "low", 1),
    Change("Registry definition update", "guidance", "high", 2),
]

rows = [(c, triage(c)) for c in ITEMS]
print(f"{'source':34s} {'kind':9s} {'impact':7s} {'q':>2s}  action")
print("-" * 70)
for c, action in rows:
    print(f"{c.source[:34]:34s} {c.kind:9s} {c.impact:7s} {c.quarter:2d}  {action}")
print("-" * 70)

by_q = {}
for c, action in rows:
    if action != "ignore":
        by_q[c.quarter] = by_q.get(c.quarter, 0) + HOURS[c.impact]
print("effort by quarter (hours):")
for q in sorted(by_q):
    print(f"  Q{q}: {by_q[q]:.1f}")
print(f"total: {sum(by_q.values()):.1f} hours across {len(by_q)} quarters")
print("act now:", sum(1 for _, a in rows if a == "act now"),
      " schedule:", sum(1 for _, a in rows if a == "schedule"),
      " watch:", sum(1 for _, a in rows if a == "watch"),
      " ignore:", sum(1 for _, a in rows if a == "ignore"))
```

Expected output:

```text
source                             kind      impact   q  action
----------------------------------------------------------------------
FDA guidance revision              guidance  high     1  act now
MDCG guidance update               guidance  high     1  act now
ICD-10-CM annual release           code      high     4  act now
LOINC release                      code      medium   2  schedule
SNOMED release                     code      medium   2  schedule
HL7 FHIR ballot                    standard  medium   3  schedule
CDISC version                      standard  low      3  ignore
ICH E9(R1) Q&A                     guidance  medium   1  schedule
Dataset version 3.2                dataset   high     2  act now
Retraction in reading list         paper     high     3  act now
New benchmark paper                paper     low      3  ignore
Society guideline                  guidance  high     4  act now
EMA guideline                      guidance  medium   2  schedule
ISO standard amendment             standard  medium   4  schedule
Journal changes policy             paper     low      1  ignore
Registry definition update         guidance  high     2  act now
----------------------------------------------------------------------
effort by quarter (hours):
  Q1: 9.5
  Q2: 12.5
  Q3: 5.5
  Q4: 9.5
total: 37.0 hours across 4 quarters
act now: 7  schedule: 6  watch: 0  ignore: 3
```

Four readings matter. The load peaks in Q2 at 12.5 hours, which is the quarter with a dataset release and two code releases; a team that has not planned for that peak will defer the work. Seven items are act-now, which is a high proportion and reflects a list weighted toward regulatory sources. No item is classified as watch, because the medium-impact non-regulatory items were labelled low; in a real list, the watch category is where most items belong. And the total of 37 hours is the effort for the changes, not for the routine, which adds detection and audit time.

The rule is deliberately simple. A more elaborate scoring model would produce a more precise number and a less consistent one, because two reviewers would weight the factors differently.

To adapt the script, replace the items with your register and adjust the hours per impact level. Keep the four actions, because the discipline is in the categories rather than in the arithmetic.

## 8. Data, standards, and regulatory context

### 8.1 What to monitor, by issuer

| Issuer | What changes | Typical cadence | Where to watch |
|---|---|---|---|
| FDA | Guidance, rules, device authorisations | Continuous | Guidance database, Federal Register |
| European Commission | Regulations, implementing acts, AI Act timeline | Continuous | EUR-Lex, Commission pages |
| MDCG | Device guidance | Several per year | MDCG guidance list |
| EMA | Medicines guidance and product information | Continuous | EMA site |
| HL7 | FHIR releases and ballots | Periodic | HL7 site |
| LOINC | Terminology releases | Twice yearly | LOINC release notes |
| SNOMED International | Terminology releases | Twice yearly | SNOMED release notes |
| CDISC | Data standards | Periodic | CDISC site |
| ICH | Efficacy and safety guidelines | Occasional | ICH site |
| FDA and Crossref | Retractions and corrections | Continuous | Crossref API, Retraction Watch |

### 8.2 Release cadence and planning

| Source | Predictable date | Planning consequence |
|---|---|---|
| ICD-10-CM | 1 October, annually | Freeze analysis code in September |
| LOINC | February and August | Re-check code mappings after release |
| SNOMED CT | January and July | Re-check concept mappings |
| US federal fiscal rules | 1 October | Align budget and compliance cycles |
| EU transition deadlines | Fixed dates | Put in the launch plan |

A predictable release date is the easiest kind of trigger to plan for, and it is the one most often missed because it falls outside the quarterly cycle.

### 8.3 Archive and record standards

| Practice | What it provides | Limit |
|---|---|---|
| Web archive snapshot | A dated copy of a page | Some pages are not archived |
| DOI | A persistent identifier | Content can change behind it |
| Document number | A stable identifier for a guidance | Not every page has one |
| Version pin | The exact version used | Requires discipline |
| Retention schedule | How long records are kept | Set by policy, not by convenience |

A record that cannot be produced on request is not a record. The archive is the evidence, and the register is the index to it.

### 8.4 What the standards do not provide

- A notification when a source changes.
- A guarantee that a DOI resolves to the same content.
- A definition of materiality for your decision.
- A replacement for reading the change.

The routine supplies those, and the standards supply the identifiers that make the routine possible.

### 8.5 Reading a release note

A release note is the cheapest early warning available, and most teams read only its headline. The note is usually structured, and a structured read takes about five minutes.

| Field in the note | What to look for | Why it matters |
|---|---|---|
| Release date | The date of publication, not the date you downloaded | The date fixes which version applies |
| Effective date | When the release applies in practice | Two dates can differ by months |
| New concepts or codes | Additions | Usually additive and low risk |
| Retired concepts or codes | Removals and replacements | The main source of silent breakage |
| Changed definitions | Revision of an existing item | A mapping can remain valid but mean something else |
| Known issues | Errata and corrections | A known issue may be fixed in a patch |
| Migration guidance | Recommended replacement | The cheapest route to a fix |
| Compatibility | Supported versions and dependencies | A release can force other upgrades |

Read the retirement list first. Additions rarely break anything, while a retirement can leave a code without a home, and the failure appears downstream as a missing value rather than as an error.

The second read is the changed-definition list. A concept that keeps its code but changes its meaning is the most dangerous change in a terminology release, because the mapping still resolves and the analysis still runs.

Record three things from every release note: the version, the date read, and the list of retirements that touch your mappings. The rest of the note can be summarised in a sentence.

## 9. Industry practice and case evidence

**QMSR transition (documented).** The FDA's Quality Management System Regulation became effective on 2 February 2026. A summary written in 2025 and reviewed quarterly in 2026 would catch the change at the first review after the effective date, and a summary reviewed annually might not. The case shows why the cadence should follow the source rather than the calendar.

**AI Act timeline (documented).** The Commission's implementation timeline has been amended, and the date for Annex I high-risk products is now 2 August 2028. A team monitoring the Act through a 2025 summary would carry the earlier date until someone checked the Commission's page. The case shows why a dated claim is easier to maintain than an undated one.

**ICD-10-CM annual release (documented).** The code set is updated on 1 October each year in the United States. A cohort definition built on the previous year's codes can change behaviour at the release, and the change is predictable. The case shows why a release calendar belongs in the maintenance plan.

**Terminology releases (documented).** LOINC and SNOMED CT publish periodic releases with new and retired concepts. A mapping built on one release can lose concepts at the next, and the loss is silent unless the mapping is re-checked.

**Retraction monitoring (documented).** Crossref carries retraction data and exposes it through its API, including records sourced from Retraction Watch. A reading list that is never checked against retraction data will eventually cite a retracted paper.

### A twelve-month calendar

| Month | Activity | Hours | Output |
|---|---|---:|---|
| January | Register review and ownership | 3 | Updated register |
| February | LOINC release check | 1 | Mapping note |
| March | Q1 quarterly review | 2 | Checklist |
| April | Regulatory deep check | 4 | Status memo |
| May | Literature alert review | 2 | Reading log |
| June | Q2 quarterly review | 2 | Checklist |
| July | SNOMED release check | 1 | Mapping note |
| August | LOINC release check | 1 | Mapping note |
| September | Q3 review and ICD-10 preparation | 3 | Checklist and freeze |
| October | ICD-10 release and cohort re-check | 4 | Re-extraction note |
| November | Regulatory deep check | 4 | Status memo |
| December | Q4 review and annual retrospective | 3 | Checklist and plan |

The total is about 30 scheduled hours plus the weekly alert triage, which brings the annual effort to the 55 hours estimated in section 5. The calendar is deliberately light in the months with predictable releases and heavier in the two deep-check months.

A team with more sources can add a monthly deep check rather than lengthening the quarterly review, because the quarterly review is a checklist and the deep check is reading.

### Documented facts and inferences

| Statement | Status |
|---|---|
| QMSR effective 2 February 2026 | Documented by FDA |
| AI Act Annex I date now 2 August 2028 | Documented on the Commission timeline |
| ICD-10-CM updates annually on 1 October | Documented by CMS and CDC |
| LOINC and SNOMED release periodically | Documented by the maintainers |
| Crossref exposes retraction data | Documented by Crossref |
| Most teams review sources annually | Inference from practice |

### What mature teams do (inference)

- They keep a register with owners and triggers.
- They check status at the issuer, not only links.
- They propagate changes to documents and models.
- They archive what they read.
- They record the cost of the routine.

### A cadence decision worked through

A team maintains a model that flags deteriorating inpatients. The model's inputs are laboratory values mapped through LOINC, its cohort is defined with ICD-10-CM, and its regulatory position depends on FDA guidance.

The team first proposes an annual review, on the grounds that nothing changed much last year. The proposal fails one test: the inputs include a terminology that is released twice a year, and a cohort definition that is released once a year, with a fixed date.

The register is rebuilt around those facts.

| Source | Cadence | Trigger | Reason |
|---|---|---|---|
| FDA guidance | Quarterly | Any Federal Register notice | The regulatory position can change without notice |
| LOINC | Each release, February and August | Release note | Mappings can break silently |
| ICD-10-CM | Each release, 1 October, plus a September freeze | Release note | Cohort definitions depend on the code set |
| Clinical guideline | Quarterly | Guideline update | The outcome definition can change |
| Vendor documentation | Each upgrade | Upgrade event | Interface behaviour can change |
| Literature | Monthly alerts | None | Signal, not obligation |

The two sources that drove the decision are the terminology and the code set, and neither is the source the team was thinking about. The regulatory source set the floor at quarterly; the releases set the trigger points.

The annual estimate that follows is 44 hours: 12 for the quarterly reviews, 8 for the two deep checks, 4 for the regulatory notices, 6 for the releases and remapping, 6 for the literature, and 8 for triage and propagation. That is within the 55-hour reference and well above the four hours the annual review implied.

The case is an inference about a plausible team rather than a documented organisation, and it is presented as a worked pattern. The numbers are illustrative and should be replaced with logged time after the first year.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Annual review only | The register says "reviewed yearly" | Cadence set by habit | A 2026 date covers a 2024 claim | Set cadence per source |
| Date taken as status | A page says "updated" | Update date confused with validity | Superseded guidance used | Check for a successor document |
| Link check only | The only test is HTTP 200 | A live page can be wrong | Confident citation of dead content | Check status at the issuer |
| Monitor list without owners | Nobody knows who reads what | Diffusion of responsibility | Items quietly dropped | One owner per row |
| Alert fatigue | Hundreds of alerts per week | No triage rule | Alerts ignored | Two filters and four actions |
| Quiet quarter read as safety | No changes found | Absence of evidence as evidence | Routine cut | Record the quiet quarter |
| Release date missed | ICD-10 change surprises the team | Calendar not consulted | Cohort breaks at launch | Freeze before release |
| Terminology drift | Codes silently unmapped | Release not re-checked | Cohort definition changes | Re-check mappings |
| Retraction unnoticed | Cited paper withdrawn | No retraction watch | Claim built on retracted work | Check at each reuse |
| Change not propagated | Summary updated, model not | Documents and code drift apart | Two truths in one team | Link register to artefacts |
| Version not pinned | "Current version" in a methods note | Floating reference | Irreproducible analysis | Record version and date |
| Guidance read from secondary summary | A consultant's slide is the source | Summary of a summary | Error amplified | Read the primary document |
| Over-monitoring | Twenty sources checked weekly | Cost exceeds benefit | Routine abandoned | Tier the sources |
| Under-monitoring | One source checked yearly | Convenience | Missed change | Minimum tier per source |
| Register unread | A file nobody opens | No use in decisions | Maintenance theatre | Review at decision points |
| Archive missing | The page changed and no copy exists | No snapshot taken | Cannot show what was relied on | Snapshot on read |
| Change misjudged | A revision treated as minor | No impact test | Claim left stale | Ask the impact questions |
| Owner left the team | Row unassigned after departure | No succession | Silent gap | Quarterly ownership check |
| Tool purchased, not used | A licence with no logins | Buying as substitute for doing | Cost without coverage | Pilot before purchase |
| Deadline misread | A transition date from memory | No primary check | Late compliance work | Check the issuer's date |

### A monthly health check

- Every row has an owner.
- Every row has a last-checked date.
- Every high-impact row has been checked within the quarter.
- Every changed item has a linked artefact update.
- Every dead link has a replacement or a retirement note.
- Every archived item has a snapshot.

A register that passes these six checks is doing its job. A register that fails more than two needs a repair before it needs more sources.

### Red flags in a maintenance plan

- The plan lists sources but not decisions.
- The cadence is uniform across very different issuers.
- There is no owner.
- There is no archive.
- The plan measures reading, not propagation.
- The plan has never been revised.
- The plan costs more than the risk it covers.

### Red flags in a maintenance habit

- Reading alerts without recording them.
- Saving links without dates.
- Trusting summaries over primary documents.
- Checking at a deadline rather than on a schedule.
- Treating a subscription as coverage.
- Treating a conference talk as a status update.

### Detection methods compared

| Method | Coverage | Cost | Lag | Best used for |
|---|---|---|---|---|
| Manual check at the issuer | Exact | Minutes per item | As long as the interval | High-impact sources |
| Email alerts from the issuer | Good where offered | Free | Hours to days | Regulators and standards bodies |
| RSS or feed reading | Medium | Low | Hours | Journals and news |
| Literature alerts | Broad | Low | Days | Signal, not obligations |
| Terminology release notes | Exact | Minutes | At release | Mappings and code sets |
| Automated link checking | Narrow | Low | Days | Dead-link detection only |
| Colleague networks | Informal | Social | Variable | Early warning, not evidence |
| Conference programmes | Partial | Attendance | Months | Orientation and early signal |

No single method covers a register. The manual check is the only one that establishes status, and the automated methods are best at telling you where to look.

### When to escalate beyond the routine

- The change contradicts a claim already in a submission.
- The change alters a cohort definition after data extraction.
- The change retires a code used in a live model.
- The change comes with a compliance date inside the project.
- The change is ambiguous and two readers disagree.

Escalation means a named person decides, in writing, whether to rework, to document the earlier basis, or to pause. The routine detects; escalation decides.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Issuers publish changes where you watch | The monitoring depends on it | Change announced elsewhere | Check the issuer's index |
| The update date is reliable | Currency assessment depends on it | Page edited without notice | Compare versions |
| A successor is discoverable | Supersession depends on it | Withdrawal with no pointer | Search the issuer |
| The register is read | The routine has effect | Nobody opens it | Check decision records |
| Owners stay in post | Continuity depends on it | Staff turnover | Quarterly ownership check |
| Impact can be judged quickly | Triage depends on it | Change is subtle | Escalate uncertain items |
| Artefacts can be updated | Propagation depends on it | Frozen model or document | Test one change path |
| Archives are admissible | Records depend on it | Policy rejects snapshots | Ask quality and legal |
| The cadence is affordable | The plan is feasible | Hours exceed capacity | Compare estimate to logged time |
| Alerts are relevant | The filters work | Broad keywords | Measure precision of alerts |
| Sources have stable identifiers | Tracking depends on it | No document number | Use URL plus title plus date |
| Release dates are published | Planning depends on it | Irregular releases | Watch the issuer's calendar |
| Terminology mappings are maintained | Cohort stability depends on it | Release retires concepts | Re-run mapping test |
| The team agrees on materiality | Triage is consistent | Different thresholds | Write the rule down |
| The issuer's index is complete | Supersession checks depend on it | Documents removed quietly | Compare against a second index |
| The archived copy is legible | Evidence depends on it | Dynamic page archived blank | Open the snapshot |
| The register survives a handover | Continuity depends on it | No written procedure | Have a colleague run one cycle |
| The routine has an owner with time | The cadence is kept | Owner at full capacity | Check logged hours |

Two limits are structural. First, no monitoring system can see a change that an issuer does not publish; a quiet issuer and a quiet source look the same. Second, the cost of monitoring rises with coverage, so a plan that covers everything is either unaffordable or unread.

A third limit is time. Maintenance competes with delivery, and a plan that assumes spare capacity will be the first thing cut when delivery is under pressure. The plan should be small enough to survive a bad quarter.

A fourth limit is authority. A reviewer can record that a guidance changed, but cannot declare what the change means for a regulated claim. That judgement belongs to the regulatory owner.

A fifth limit is visibility. A change that is announced in a webinar, a press release, or a conversation is not visible to a link checker, and the register will not record it. The routine covers published sources, and the team needs a second, social channel for everything else.

A sixth limit is interpretation. The same release note can be read as a minor addition by one team and as a material change by another, and both readings can be defensible. The register reduces the disagreement by recording the reasoning at the time rather than reconstructing it later.

A seventh limit is that the routine is evidence about the routine. It shows that a check happened and what was found; it does not show that the source was correct, that the team understood it, or that the resulting artefact was updated well. Those are separate quality questions.

## 12. Exercises

**Exercise 1 — Build the register (application, 40 minutes).**
Take ten sources you rely on and build a register with owner, cadence, trigger, impact level, and last-checked date.
Deliverable: a ten-row table plus one sentence on the decision each source informs.
Acceptance criteria: every row has an owner and a cadence; at least two rows have a trigger date; the register names the artefact each source feeds.
Time: 40 minutes.

**Exercise 2 — Cadence decision (judgement, 25 minutes).**
Assign each register row to weekly, monthly, quarterly, or release-triggered monitoring and justify the choice.
Deliverable: a cadence column with a one-line rationale per row.
Acceptance criteria: at least one row is release-triggered; no row uses "annually" without a reason; high-impact rows are checked at least quarterly.
Time: 25 minutes.

**Exercise 3 — Triage the backlog (analysis, 35 minutes).**
Take twenty changes and classify each as act now, schedule, watch, or ignore using the rule in section 5.
Deliverable: a classified list with hours per action.
Acceptance criteria: at least three items land in each of act now and ignore; the total hours are stated; one borderline item is discussed.
Time: 35 minutes.

**Exercise 4 — Detect a supersession (research, 30 minutes).**
Choose a guidance you cite and establish whether it is current, superseded, or withdrawn, from the issuer's own pages.
Deliverable: a status note with the evidence and the successor document if one exists.
Acceptance criteria: the issuer's page is cited, not a summary; the retrieval date is recorded; the result is one of three explicit states.
Time: 30 minutes.

**Exercise 5 — Propagate a change (application, 30 minutes).**
Take one changed source and trace every artefact that depends on it: document, model, mapping, slide.
Deliverable: a dependency list with the update required for each.
Acceptance criteria: at least three artefacts are found; each has an owner; one update is completed.
Time: 30 minutes.

**Exercise 6 — Cost the routine (calculation, 25 minutes).**
Estimate the annual hours and cost of your register using the method in section 5.
Deliverable: a table of cadence, items, minutes per item, and annual hours, plus a cost at a stated rate.
Acceptance criteria: the arithmetic is shown; the total is compared to the 55-hour reference; one activity is cut or deferred with a reason.
Time: 25 minutes.

**Exercise 7 — Retire a source (critique, 20 minutes).**
Identify one source in your list that should be retired and write the retirement note.
Deliverable: a short note with the reason, the replacement, and the residual risk.
Acceptance criteria: the reason is one of superseded, unreachable, unreliable, or no longer relevant; the replacement is named; the archive is referenced.
Time: 20 minutes.

**Exercise 8 — Design the twelve-month calendar (planning, 35 minutes).**
Build a calendar with predictable releases, quarterly reviews, and two deep checks, with hours per month.
Deliverable: a twelve-row calendar and an annual total.
Acceptance criteria: the total is between 25 and 60 hours; the heaviest month is justified; at least three release dates are sourced.
Time: 35 minutes.

### How to run the exercises

- Keep one folder per exercise with the register, the evidence, and the note.
- Record the date of every retrieval, not only the date of the exercise.
- Ask a colleague to check the two exercises that rest on judgement.
- Compare your register with the reference register after each exercise.
- Keep the failed checks; they are the useful part of the record.

### Marking rubric

| Criterion | Weak | Adequate | Strong |
|---|---|---|---|
| Evidence | Assertion only | One source | Primary source with date |
| Reasoning | Conclusion stated | Reason given | Alternative considered |
| Record | No artefact | Table or note | Dated, attributable record |
| Propagation | Not considered | Named | Named, owned, updated |
| Cost | Ignored | Estimated | Estimated and compared |

## 13. Self-check answers

**Answer 1.** A register with owner, cadence, trigger, impact, and last-checked date; each row should name the decision or artefact it informs. Without that link the register becomes a reading list.

**Answer 2.** High-impact regulatory sources are checked at least quarterly and at every release; terminology is checked at each release; low-impact news sources can be monthly or dropped. "Annually" is defensible only where the issuer publishes annually.

**Answer 3.** Act now: the change invalidates a current claim or a live cohort. Schedule: the change will matter before the next review. Watch: the change may matter, but not yet. Ignore: no path from the change to a decision. The classes are about consequence, not about how interesting the change is.

**Answer 4.** Current means the issuer lists it as in force with no successor. Superseded means a successor exists and applies. Withdrawn means the issuer has removed it and it should not be relied on. A page that simply disappears is not evidence of withdrawal.

**Answer 5.** The dependent artefacts are usually a written summary, a slide deck, a model specification, a code mapping, and a validation report. Most teams find the summary and miss the mapping.

**Answer 6.** Multiply items by checks per year by minutes per check, divide by sixty for hours, and multiply by a loaded rate for cost. A register of forty items checked quarterly at ten minutes is about 27 hours before triage.

**Answer 7.** Retire when a source is superseded with no residual use, unreachable with no archive, unreliable by provenance, or no longer relevant to any decision. Keep the record and note the replacement; deleting the entry loses the history.

**Answer 8.** A workable calendar spreads predictable releases across the year, places quarterly reviews in March, June, September, and December, and puts deep checks in the two months with the largest regulatory exposure. The total should survive a busy quarter, which is why 25 to 60 hours is the sensible band.

### Common wrong answers

| Answer | Why it fails | Better answer |
|---|---|---|
| "The list is reviewed annually" | Cadence set by habit | Cadence set per source |
| "The link works" | Link check is not a status check | Status checked at the issuer |
| "It says updated recently" | Date taken as status | Successor searched for |
| "Nobody has complained" | Absence of evidence | Quiet quarters recorded |
| "The subscription covers it" | Access mistaken for use | Reading and propagation logged |
| "We will catch it at the next review" | Interval too long for the risk | Trigger or shorter cadence |

## 14. Interview questions

**1. How do you know your sources are current?**
A strong answer names a register, a cadence, and a check at the issuer. A weak answer says the list was reviewed last year.

**2. What is the difference between a link check and a status check?**
A link check asks whether the page loads. A status check asks whether the document is still in force, and it requires the issuer's index.

**3. How would you decide how often to check a source?**
By the cost of missing a change and the issuer's release cadence, not by convenience. High-impact and fast-moving sources get the shortest interval.

**4. What do you do when a guidance is superseded mid-project?**
Record the date, assess the impact on the claim, and decide whether to rework or to document the earlier basis. The decision is recorded either way.

**5. How do you avoid alert fatigue?**
Two filters, a small set of high-signal sources, and a triage rule that permits ignoring most items. The failure mode is a subscription that nobody reads.

**6. How do you handle a terminology release that retires codes?**
Re-run the mapping, list the affected concepts, and decide whether to remap, freeze the old release, or change the definition. The decision is documented.

**7. What evidence would you show an auditor that your sources were current?**
The register with dates, the archived snapshots, and the change log linking each update to an artefact.

**8. How do you keep maintenance from being cut?**
By costing it, recording quiet quarters, and tying it to decisions that the organisation already cares about. Maintenance with no visible output is the first thing cut.

**9. What is the weakest part of a maintenance routine?**
Propagation. Detection is cheap and visible; updating everything that depended on the source is where routines fail.

**10. How would you onboard a colleague into the routine?**
Give them one row with an owner, a cadence, and a decision attached, and have them run one quarterly review with you. The routine is learned by doing one loop.

### What interviewers listen for

| Signal | Weak answer | Strong answer |
|---|---|---|
| Specificity | "We keep an eye on it" | A named register and cadence |
| Evidence | "It is standard practice" | A date and a source |
| Consequence | "It is good practice" | The decision that would change |
| Ownership | "The team does it" | A named owner and a handover |
| Cost | "It does not take long" | Hours per quarter |
| Failure | "We have not missed anything" | A near miss and its fix |

## 15. Key terms

- **Register** — the table of sources, owners, cadences, and status.
- **Cadence** — how often a source is checked.
- **Trigger** — an event that forces a check outside the cadence.
- **Triage** — classifying a change by consequence and effort.
- **Materiality** — whether a change affects a decision.
- **Supersession** — replacement of a document by a successor.
- **Withdrawal** — removal of a document by its issuer.
- **Currency** — whether a source still applies.
- **Status check** — verification at the issuer, not the link.
- **Snapshot** — a dated archived copy of a page.
- **Provenance** — the origin and custody of a source.
- **Propagation** — updating artefacts that depend on a source.
- **Dependency** — a document, model, or mapping that relies on a source.
- **Release** — a published version of a standard or terminology.
- **Release note** — the document describing what changed in a release.
- **Deprecation** — a maintainer's warning that an item will be removed.
- **Retirement** — removing a source from the active list.
- **Archive** — the stored record of what was read and when.
- **Change log** — the record of updates made in response to changes.
- **Alert** — an automated notification of a possible change.
- **Watch list** — the set of sources monitored routinely.
- **Impact level** — the consequence of missing a change.
- **Owner** — the person accountable for a register row.
- **Deep check** — a slower review that reads the primary documents.
- **Quarterly review** — the periodic checklist over the whole register.
- **Backlog** — changes detected but not yet triaged.
- **False alert** — a notification that does not require action.
- **Precision of alerts** — the share of alerts that matter.
- **Version pin** — recording the exact version used.
- **Retention** — how long records are kept.
- **Audit trail** — the evidence that checks happened.
- **Document number** — a stable identifier for a guidance.
- **Floating reference** — a citation to "the current version" without a date.
- **Status page** — the issuer's page that states whether a document is in force.
- **Erratum** — a published correction to a document.
- **Patch release** — a small release fixing defects.
- **Sunset date** — the date a version stops applying.
- **Transition period** — the window during which an old rule still applies.
- **Conformity assessment** — the process of demonstrating compliance.
- **Notified body** — an organisation assessing conformity under EU rules.
- **Predicate** — a previously cleared device used as a comparison.
- **Change control** — the procedure for approving and recording changes.
- **Impact assessment** — the analysis of what a change affects.
- **Review cycle** — the interval between scheduled reviews.
- **Escalation path** — who decides when the routine is not enough.

## 16. Sources

All URLs were checked on 26 October 2026. Issuers change pages without notice; re-check at the source before relying on a date.

**US Food and Drug Administration, guidance search.** https://www.fda.gov/regulatory-information/search-fda-guidance-documents
Good for: the authoritative list of FDA guidance documents and their status. Not for: interpretation of how a guidance applies to your device.

**US Food and Drug Administration, Quality Management System Regulation.** https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
Good for: the QMSR text and the 2 February 2026 effective date. Not for: the parallel ISO 13485 text, which is purchased separately.

**Federal Register.** https://www.federalregister.gov/
Good for: rules, notices, and the legal date of US regulatory actions. Not for: explanatory guidance, which appears elsewhere.

**EUR-Lex, Artificial Intelligence Act.** https://eur-lex.europa.eu/eli/reg/2024/1689/oj
Good for: the consolidated legal text of Regulation (EU) 2024/1689. Not for: the implementation timeline, which is maintained by the Commission.

**European Commission, AI Act policy page.** https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
Good for: application dates and implementation notes, including the revised Annex I date. Not for: the legal text itself.

**Medical Device Coordination Group guidance.** https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en
Good for: MDCG-endorsed guidance on MDR and IVDR application. Not for: national competent authority decisions.

**European Medicines Agency.** https://www.ema.europa.eu/en
Good for: medicines guidance, product information, and committee outcomes. Not for: device regulation.

**HL7 FHIR specification.** https://hl7.org/fhir/
Good for: the current FHIR release and its change history. Not for: local implementation profiles.

**LOINC downloads and releases.** https://loinc.org/downloads/
Good for: release files, release notes, new and retired codes, and the release cadence. Not for: mapping decisions in your own system.

**SNOMED International.** https://www.snomed.org/
Good for: release information and terminology governance. Not for: the licensing terms of a member country.

**CDISC standards.** https://www.cdisc.org/standards
Good for: current data standards and implementation guides. Not for: regulator-specific acceptance of a submission.

**International Council for Harmonisation.** https://www.ich.org/page/efficacy-guidelines
Good for: efficacy and safety guidelines, including estimands. Not for: regional implementation.

**Centers for Disease Control and Prevention, ICD-10-CM.** https://www.cdc.gov/nchs/icd/icd-10-cm.htm
Good for: the annual code set and its release schedule. Not for: payment rules, which sit with CMS.

**Crossref REST API.** https://www.crossref.org/documentation/retrieve-metadata/rest-api/
Good for: programmatic metadata, DOI resolution, and retraction data. Not for: the editorial reasoning behind a retraction.

**Retraction Watch database.** https://retractionwatch.com/retraction-watch-database-user-guide/
Good for: retraction records and their reasons, including via Crossref. Not for: a guarantee of completeness.

**Internet Archive Wayback Machine.** https://web.archive.org/
Good for: dated snapshots of pages that may change or disappear. Not for: evidence that a snapshot was the version relied on.

**PubMed.** https://pubmed.ncbi.nlm.nih.gov/
Good for: publication status, errata, and retraction notices linked to a record. Not for: the full text of paywalled articles.

**International Organization for Standardization, ISO 13485:2016.** https://www.iso.org/standard/59752.html
Good for: the standard catalogue entry and its status. Not for: the text, which must be purchased.

**TRIPOD+AI statement.** https://www.tripod-statement.org/
Good for: current reporting guidance for prediction models. Not for: study quality assessment, which is PROBAST.

**PROBAST.** https://www.probast.org/
Good for: risk-of-bias assessment for prediction models. Not for: reporting completeness.

**EQUATOR Network.** https://www.equator-network.org/
Good for: finding the reporting guideline for a study type. Not for: enforcing it.

**Centers for Medicare and Medicaid Services, ICD-10.** https://www.cms.gov/medicare/coding-billing/icd-10-codes
Good for: payment-related code updates and the US release schedule. Not for: clinical coding guidance, which sits with CDC.

**FDA, artificial intelligence-enabled medical devices.** https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
Good for: the agency's stated position and the list of authorised AI-enabled devices. Not for: a complete authorisation history of a specific product, since the list is explicitly not comprehensive.

**HL7 International.** https://www.hl7.org/
Good for: ballot schedules, implementation guides, and working group material. Not for: the normative FHIR text alone, which is at hl7.org/fhir.

**ClinicalTrials.gov.** https://clinicaltrials.gov/
Good for: trial registration status, results posting, and protocol amendments. Not for: publication status.

**OpenAlex.** https://openalex.org/
Good for: programmatic literature tracking, citation links, and venue metadata. Not for: authoritative retraction status, which is better checked at the publisher or Crossref.

### Verification status of this list

| Status | Count | Note |
|---|---:|---|
| Issuer's own page, checked 26 October 2026 | 26 | Primary source for the claim |
| Secondary page used only for orientation | 0 | None used |
| Unverified | 0 | Removed rather than listed |

Every entry above resolves to the organisation or document named, and every URL returned a page at check. Two issuers block automated requests, so those pages were opened in a browser rather than checked by script.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_communities-and-continuing-education.md`](08_communities-and-continuing-education.md)
- How to read this list: [`01_how-to-read-this-list.md`](01_how-to-read-this-list.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 20, career: [`../20_career/`](../20_career/)
- Module 15, evaluation and monitoring: [`../15_mlops/`](../15_mlops/)
- Module 02, data landscape: [`../02_data_landscape/`](../02_data_landscape/)
- Module 19, capstone: [`../19_capstone/`](../19_capstone/)

### Where each part of the routine is taught

| Part of the routine | Where it is developed |
|---|---|
| Reading a source critically | Topic [`01_how-to-read-this-list.md`](01_how-to-read-this-list.md) |
| Checking a citation's status | Topic [`02_supplied-clinical-library.md`](02_supplied-clinical-library.md) |
| Tracking regulatory change | Topic [`03_regulation-and-policy.md`](03_regulation-and-policy.md) |
| Re-checking a method | Topic [`04_methods.md`](04_methods.md) |
| Following a definition's era | Topic [`05_clinical-background.md`](05_clinical-background.md) |
| Re-checking a dataset version | Topic [`06_datasets-and-benchmarks.md`](06_datasets-and-benchmarks.md) |
| Following a market claim | Topic [`07_industry-and-market.md`](07_industry-and-market.md) |
| Learning from a community | Topic [`08_communities-and-continuing-education.md`](08_communities-and-continuing-education.md) |

The routine is last because it applies to all of the others. Each topic's list is a snapshot with a date, and this topic is the procedure that keeps the dates honest.

A final word on discipline. The difference between a source list that stays useful and one that quietly rots is rarely effort; it is a small number of habits kept without exception: record the date, check at the issuer, archive what you read, and update what depended on it.

The routine is the last topic because it is the one that determines whether the other eight stay true. A source list is a snapshot; a maintenance routine is what keeps the snapshot from becoming a historical document.
