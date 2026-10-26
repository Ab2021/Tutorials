# Judging

> **Module 19 — Capstone, Hackathons, and Competitions**
> Teaching time: 3.5 hours
> Prerequisites: Topics 01, 04, 09
> Version 2026.3

## 1. Why this topic exists

A capstone is judged by a panel of three to five people who have read the dossier, watched the
demo, and listened to the defense.
The panel's job is not to rank enthusiasm. It is to decide whether the evidence supports the claim,
whether the team understands its own work, and whether the artefact should exist.
That decision is made with a rubric, and the rubric is a measurement instrument. Like any
instrument, it can be valid, reliable, and fair, or it can produce a number that looks precise and
means very little.

Teams experience judging as a black box, and that is a design failure rather than an inevitability.
A panel that publishes its criteria, calibrates on anchor submissions, and records its reasoning
gives teams something to aim at. A panel that scores from impression gives teams a lottery.
The difference is visible in the scores: a calibrated panel produces rankings that survive a change
of judge, and an uncalibrated one produces rankings that change when a judge is replaced.

The stakes are not only grades.
In industry, the same structure appears in funding panels, procurement committees, regulatory
review, and internal model-approval boards. A data scientist who has served on a calibrated panel
knows why a criterion needs an anchor, why a lenient judge distorts a ranking, and why a conflict
of interest must be declared before the discussion rather than after the vote.
Those are transferable skills, and they are rarely taught explicitly.

There is a second reason to study judging: it improves the work.
A team that understands the rubric stops polishing the demo and starts fixing the validation
report, because the weights say where the points are.
That is not gaming the rubric; it is reading the specification. A rubric that rewards evidence
should change behaviour toward evidence, and a rubric that rewards presentation will do the
opposite, which is why the design of the rubric is itself an ethical choice.

This topic covers rubric design, calibration between judges, the standard scoring failures, and the
arithmetic of deductions and individual multipliers.
It includes a calibration exercise with anchor submissions, because calibration is a practice rather
than a concept and cannot be learned from a description.

One more reason to study judging: every capstone team is also a judge of its own work.
The panel's rubric is the instrument the team should use before submission, and the standard
failures are the ones a team commits when it reviews its own dossier: halo from effort, leniency
toward its own choices, and a holistic impression that hides a weak section.
A team that can score its own work against anchors will fix the validation report before the panel
does.

The topic is written for two audiences at once. A student reads it to understand what the panel
will reward and to self-assess. A future panel member reads it to design an instrument that
produces a defensible decision. The same concepts serve both, which is why the treatment is more
detailed than a student strictly needs.

A final framing: judging is a service to the teams, not a verdict on them.
A panel that explains its criteria, applies them consistently, and gives criterion-level feedback
has done something more useful than ranking. It has told a team what good work looks like in the
terms the profession uses, which is the part of an assessment that lasts after the grade is
forgotten.

That is also the reason the rubric is published to students before the work begins. A criterion
hidden until the end is a trap; a criterion published at the start is a specification, and a
specification is something a team can meet.

A team that has read this topic should be able to do three things on the day: predict its own score,
explain a criterion-level gap without appealing to effort, and ask a question about a score that
the panel can answer from the anchor. Those three behaviours are what a well-run panel makes
possible.

## 2. Learning outcomes

By the end of this topic you can:

1. **Design** an analytic rubric with criteria, weights, and behavioural anchors at each level.
2. **Distinguish** an analytic rubric from a holistic impression and explain when each is
   appropriate.
3. **Calibrate** a panel on anchor submissions and measure the disagreement before scoring live
   work.
4. **Compute** weighted totals, judge-level severity, and normalised scores, and explain what
   normalisation does and does not fix.
5. **Identify** halo, leniency, severity, central tendency, order effects, and affinity bias in a
   score set.
6. **Apply** deductions and an individual multiplier consistently, and document the reasoning.
7. **Run** a panel meeting that reaches a defensible decision within its time budget.
8. **Give** feedback that a team can act on, tied to criteria rather than to impressions.

Outcomes 1 to 3 are assessed in Exercises 1, 2, and 6, which together are a calibration session.
Outcomes 4 and 5 are assessed in Exercises 3 and 4 and appear in the panel's own quality review.
Outcomes 6 and 7 are assessed in Exercises 5 and 8 and are tested when a panel member is asked to
justify a deduction.
Outcome 8 is assessed in Exercise 7.

The outcomes are ordered by the sequence of a panel's work: design, calibrate, score, decide,
explain. A panel that cannot design a rubric cannot calibrate one, and a panel that cannot calibrate
will score by impression.

## 3. The core mental model

Judging is a measurement pipeline with five stages.
The rubric defines what is measured.
The anchors define what each score level looks like.
Independent scoring produces the raw observations.
Calibration and adjudication reduce disagreement.
The decision rule converts scores into a ranking and a recommendation.

```text
  CRITERIA  --->  ANCHORS  --->  INDEPENDENT  --->  CALIBRATION  --->  DECISION
  (what)          (what each      SCORING           (reduce            (rank +
                   level looks     (no talking)      disagreement)      recommendation)
                   like)
     |               |                |                  |                 |
   weights         examples         per judge          discussion        deductions
   published       of 2, 3, 4       per criterion      on outliers       multiplier
                                                                          documented
```

Read the pipeline left to right.
A failure at any stage propagates: unclear criteria produce inconsistent anchors, inconsistent
anchors produce unreliable scores, unreliable scores make calibration a negotiation, and a
negotiated score produces a decision that cannot be defended.

Three properties of the model matter.

**The rubric is a claim about what matters.** Weights are not neutral. A rubric that gives 40
percent to presentation and 10 percent to validation will produce confident teams with weak
evidence, and the teams will learn the weights faster than they learn the statistics.

**Reliability is not validity.** A panel can agree consistently on the wrong thing. High
inter-rater agreement on a criterion that does not measure the claim produces a reliable ranking of
the wrong quality. Calibration improves reliability; only rubric design improves validity.

**Disagreement is information.** A large spread between judges on one team is a signal that the
criterion is ambiguous or that the judges saw different things. Averaging it away hides the signal;
discussing it produces either a better score or a better rubric.

A fourth property is procedural.
The order of operations matters: score independently, then discuss, then decide.
A panel that discusses first and scores afterwards has produced a consensus impression, not three
independent measurements, and the spread that would have revealed disagreement is gone.

### Why weights are the ethical centre of the rubric

A weight is a statement about what the programme is willing to trade.
Giving validation 15 percent and presentation 20 says that a team can compensate for weak evidence
with a strong demo, whether or not the panel intends that.
Teams read weights accurately and allocate effort accordingly, so the weights shape the work as
well as the score.

The practical test is to ask, for each criterion, what a team would have to give up to gain a level.
If the answer is "a week of validation", the weight is telling teams to skip validation, and no
amount of exhortation will overcome it.

This is why the rubric is published before the work is done and reviewed after. It is also why a
panel should be suspicious of a rubric that has never changed: the work changes, and the instrument
should follow.

### The rubric as a specification

A published rubric does three things at once. It specifies what the programme values, it tells teams
where to spend effort, and it gives the panel a shared language for the decision.
A rubric that does only the third is a scoring form; a rubric that does all three is a teaching
instrument.

The test of whether a rubric is a specification is whether a team could predict its own score. A
team that has read the anchors and reviewed its dossier should be able to say "we are at 3 on
validation because the slope is missing" before the panel does. When teams cannot predict, the
anchors are too vague for the rubric to function as a specification.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Rubric | A structured scoring instrument with criteria and levels | A list of topics | Criteria and anchors |
| Analytic rubric | Separate scores per criterion, combined by weights | A holistic score with subheadings | Criteria x levels |
| Holistic rubric | One overall score based on impression | An analytic rubric summed by eye | Single scale |
| Criterion | A dimension of performance | A topic area | Named dimension |
| Anchor | A description or example of performance at a level | A number without a description | Level descriptor |
| Weight | The share of the total assigned to a criterion | The maximum score | Proportion summing to 1 |
| Scale | The range of possible scores per criterion | The total score | 1-5 or 0-4 |
| Calibration | Aligning judges on anchor submissions before live scoring | A discussion of expectations | Exercise and agreement |
| Inter-rater reliability | Consistency of scores across judges | Agreement on the final rank | ICC or kappa |
| Halo effect | One salient impression contaminating all criteria | Being generous | Correlation across criteria |
| Leniency | A judge who scores systematically high | A judge who liked the work | Mean score relative to panel |
| Severity | A judge who scores systematically low | A judge who disliked the work | Mean score relative to panel |
| Central tendency | Avoiding extreme scores | Being fair | Reduced variance |
| Restriction of range | Scores clustered in a narrow band | Consistency | Standard deviation |
| Order effect | Earlier teams influencing later scores | Fatigue | Position in sequence |
| Hindsight bias | Judging a decision by its outcome rather than its process | Being results-oriented | Score shift after outcome |
| Affinity bias | Favouring teams similar to the judge | Rapport | Score gap by similarity |
| Normalisation | Adjusting scores for judge severity | Changing the result | Z-scores or percentiles |
| Z-score | Score expressed in standard deviations from a mean | A transformed grade | Standard units |
| Deduction | A fixed penalty for a defined finding | A subjective penalty | Points |
| Multiplier | A factor applied to an individual's score | A group penalty | 0 to 1 |
| Conflict of interest | A relationship that could bias judgement | Any acquaintance | Declared and managed |
| Blind review | Scoring without identifying information | Anonymous submission | Redaction |
| Adjudication | A process for resolving score disagreement | Averages | Decision rule |
| Decision rule | The rule that converts scores into a decision | The highest score wins | Thresholds and ranking |
| Score sheet | The form on which a judge records scores | The final spreadsheet | Form and version |
| Panel chair | The person who manages process and conflicts | The most senior judge | Role |
| Moderation | Comparing distributions across panels | Changing scores | Mean and range |
| Pass threshold | The score at which the decision changes | The mean of the cohort | Points |
| Criterion-level feedback | Comments tied to a named criterion | General praise | Note per criterion |
| Consensus | A shared conclusion | A shared standard | Decision label |
| Agreement | A shared standard across judges | Consensus | Calibration outcome |
| Non-discriminating criterion | A criterion that scores all teams the same | A criterion everyone passes | Variance across teams |

The last term is the one a panel should track across cycles. A criterion with no variance is not
measuring a difference, and a rubric full of such criteria produces a ranking driven by the few
criteria that do vary, which may not be the ones the programme intended.

The vocabulary separates the instrument from the judgement.
A rubric is the instrument; a score sheet is a use of it; a decision record is the judgement
informed by it. A panel that conflates the three will treat a score as a verdict, and a team will
treat a verdict as a personality judgement.

A second distinction matters for teams: a criterion score is not a comment on effort. A team can
work extremely hard and score 2 on validation because the report omits the slope, and the score is
about the report rather than about the team.

A third distinction is between consensus and agreement. Consensus is a shared conclusion; agreement
is a shared standard. A panel needs the second and does not need the first, because independent
scores that differ within a level are evidence that the standard is being applied, not that the
panel is divided.

The words in the table are worth using precisely in a decision record. "The panel agreed" is
ambiguous; "the panel applied the anchor and recorded one dissent" is not.

That precision is also what makes an appeal possible. A team can challenge a score that cites an
anchor; it cannot challenge a decision that cites a feeling.

The most consequential definition is the anchor.
A rubric without anchors is a list of words, and words mean different things to different judges.
"Validation is adequate" is not an anchor; "the report states the split, gives the primary metric
with an interval, shows calibration, and names the subgroups" is.

A second definition matters for fairness: normalisation.
Normalisation adjusts for a judge's average severity, and it does not fix a judge who is
inconsistent, biased, or scoring the wrong construct. It is a correction for a known offset, not a
cure for a broken instrument.

## 5. The mechanics, step by step

The procedure has sixteen steps, split between design (before the event) and operation (on the
day). It assumes a panel of three to five and a set of submissions.

### Design

#### Step 1. State the decision the panel is making

**Input:** the programme's assessment purpose.

**Output:** one sentence: the panel decides whether each capstone meets the evidence bar, and
ranks those that do.

**Rationale:** A rubric built for ranking differs from one built for a pass/fail decision. Ranking
requires discrimination across the whole range; a threshold decision requires accuracy near the
threshold. Confusing the two produces a rubric that is precise where it does not matter.

**Branch point:** if the purpose includes developmental feedback, add a comment field per criterion
and protect time for it. Feedback is not a by-product of scoring.

#### Step 2. Choose the criteria from the claim

**Input:** the decision and the evidence bar from Topic 04.

**Output:** six to nine criteria, each named as a quality of the work rather than a section of the
dossier.

**Rationale:** Criteria should follow the argument: decision and scope, data provenance, validation
design, modelling rigour, calibration and equity, governance, system and monitoring, value, and
reproducibility. A criterion per dossier chapter produces a rubric that rewards completeness rather
than quality.

**Branch point:** if two criteria always move together in pilot scoring, merge them. Correlated
criteria double-count the same quality.

#### Step 3. Set the weights and publish them

**Input:** the criteria and the programme's priorities.

**Output:** weights that sum to one, published before submissions.

**Rationale:** Weights encode what the programme values. Publishing them lets teams allocate effort
and lets the panel defend its priorities. A rubric with unpublished weights invites the suspicion
that the ranking was decided first.

**Branch point:** if a criterion is a knockout rather than a weight, make it a knockout explicitly.
Mixing knockouts into weights hides a fatal flaw inside an average.

#### Step 4. Write behavioural anchors for each level

**Input:** the criteria and examples of past work.

**Output:** for each criterion, four or five level descriptors that describe observable features.

**Rationale:** Anchors are what make scoring comparable. They should describe what a reader can see,
not how good the work feels. "Names the split, reports the metric with an interval, and shows
calibration" is observable; "good validation" is not.

**Branch point:** if a level descriptor cannot be observed in the artefact, rewrite it until it can.

#### Step 5. Select anchor submissions

**Input:** two or three past submissions spanning the range.

**Output:** a set of anchor cases with agreed scores and a written rationale for each.

**Rationale:** Anchors calibrate the panel on real work rather than on abstractions. They also
expose disagreements about the rubric itself, which is cheaper to fix before the event.

**Branch point:** if the panel cannot agree on an anchor case, the criterion is ambiguous. Fix the
anchor description before scoring live work.

#### Step 6. Train the panel on the rubric, not on the teams

**Input:** the rubric, the anchors, and the scoring form.

**Output:** a ninety-minute session in which each judge scores the anchors independently and then
discusses.

**Rationale:** Training reduces halo and severity drift, and it gives the panel a shared vocabulary.
It does not eliminate disagreement, and it should not; the goal is disagreement about the work
rather than about the words.

**Branch point:** if a judge's anchor scores differ from the panel by more than one level on two or
more criteria, discuss the anchors before the event rather than after.

### Operation

#### Step 7. Score independently before any discussion

**Input:** the artefacts and the scoring form.

**Output:** one score per criterion per judge, submitted before the panel meets.

**Rationale:** Independence is what makes the scores measurements. A panel that discusses first
produces a shared impression, and the spread that would reveal ambiguity disappears.

**Branch point:** if a judge has not submitted before the meeting, the judge scores in the room
before the discussion starts, not during it.

#### Step 8. Check the distributions before computing totals

**Input:** the score matrix.

**Output:** each judge's mean and standard deviation, and each team's spread across judges.

**Rationale:** Distribution checks reveal severity, central tendency, and restriction of range
before those effects contaminate the ranking. A judge with a mean half a point below the panel is
not necessarily wrong, and the panel should know before it decides.

**Branch point:** if one judge's standard deviation is less than half the panel's, the judge may be
avoiding extremes. Discuss before normalising.

#### Step 9. Discuss outliers, not everything

**Input:** the distribution check.

**Output:** a list of teams where the spread across judges exceeds one level on any criterion.

**Rationale:** Discussing every score takes hours and produces anchoring. Discussing only the
outliers focuses the panel's time on the cases where the decision could change.

**Branch point:** if more than a third of scores are outliers, the rubric is the problem. Stop and
fix the anchors.

#### Step 10. Adjudicate with a rule, not by averaging in the room

**Input:** the outlier list and the judges' reasoning.

**Output:** a recorded decision: keep the scores, revise a score with a stated reason, or mark the
criterion as unreliable for that team.

**Rationale:** Adjudication is a judgement about evidence, and it needs a rule so that it is
applied consistently. The common rule is that a score changes only if a judge identifies evidence
the others missed, not if a judge feels strongly.

**Branch point:** if the panel cannot agree, record both scores and the reason, and use the median
for the decision. A recorded disagreement is more honest than a forced consensus.

#### Step 11. Compute weighted totals

**Input:** the adjudicated scores and the weights.

**Output:** a total per team per judge, and a panel mean with a range.

**Rationale:** The arithmetic is trivial and the reporting is not. A total without a range hides
the disagreement; a total with a range tells the reader how settled the decision is.

**Branch point:** if two teams are within the smallest detectable difference implied by the spread,
report a tie rather than a ranking.

#### Step 12. Apply deductions

**Input:** the findings list and the deduction schedule.

**Output:** a reduced score per finding, with the finding recorded.

**Rationale:** Deductions handle failures that are not on the quality scale: fabricated results,
identifiable data in an artefact, no baseline, undisclosed test-set use. They must be defined in
advance and applied uniformly, or they become a way to punish a team the panel dislikes.

**Branch point:** if a finding is not on the schedule, do not invent a deduction. Record it as a
comment and decide at the panel level whether the schedule needs changing for the next cycle.

#### Step 13. Apply the individual multiplier with evidence

**Input:** the individual defense scores and the contribution log.

**Output:** a multiplier per member, with written justification and a review route.

**Rationale:** The multiplier protects the credential and the team. It must be bounded, evidenced,
and reviewable; otherwise it becomes a popularity contest.

**Branch point:** if the multiplier would change a pass to a fail, require a second reviewer. A
high-stakes individual decision should not rest on one panel member's impression.

#### Step 14. Make the decision and record the reasoning

**Input:** the totals, the deductions, and the multipliers.

**Output:** a decision record per team: decision, score, rank, key strengths, key gaps, and one
recommendation.

**Rationale:** The record is what makes the decision reviewable and what makes the feedback
specific. A decision without a record cannot be appealed, moderated, or learned from.

**Branch point:** if the panel's decision differs from the arithmetic ranking, record why. A
justified override is legitimate; an unexplained one is not.

#### Step 15. Deliver feedback against the criteria

**Input:** the decision record.

**Output:** a short written feedback note per team, with one strength, one gap, and one action per
criterion that scored below competent.

**Rationale:** Feedback tied to criteria is actionable; feedback tied to impression is not. The note
also demonstrates that the panel applied the rubric rather than its taste.

**Branch point:** if a team disagrees with a score, the appeal route is the criterion and the
anchor, not the judge's overall impression.

#### Step 16. Review the rubric after the cycle

**Input:** the score distributions and the panel's notes.

**Output:** a list of criteria that failed to discriminate, anchors that were ambiguous, and weights
that produced unintended behaviour.

**Rationale:** A rubric is a working instrument and improves with use. The review is where the panel
learns that a criterion scored identically for every team, or that a weight rewarded presentation
over evidence.

**Branch point:** if a criterion did not discriminate in two consecutive cycles, remove it or
rewrite it. A criterion that everyone passes is not measuring anything.

### Step 17. Moderate across panels

**Input:** the score distributions from two or more panels.

**Output:** a comparison of means, ranges, and pass rates, and a record of any panel whose
distribution differs materially.

**Rationale:** A programme with several panels can produce different standards for similar work. A
panel whose mean is half a point above the others is either seeing stronger work or applying
looser anchors, and the programme cannot tell without the comparison.

**Branch point:** if one panel's distribution differs by more than half a standard deviation from
the others, review its anchor scores before releasing results. The fix is calibration, not
rescaling after the fact.

### Step 18. Publish the outcome and the rubric revision

**Input:** the decision records and the post-cycle review.

**Output:** a summary of the decisions, the rubric with any changes, and a note on what changed and
why.

**Rationale:** Publishing the outcome makes the process accountable and gives the next cohort a
better instrument. It also signals that the rubric is a working document rather than a fixed rule.

**Branch point:** if the rubric changes, apply the change to the next cycle rather than to the
current one. Changing the rules after scoring is the fastest way to lose the panel's credibility.

### Why sixteen steps

The procedure is long because judging is an instrument with several failure points, and each step
addresses one of them. A panel that skips the design steps will spend the meeting arguing about
words. A panel that skips the operation steps will produce a ranking that cannot be defended.

The steps are also an order of operations. Scoring before discussion, checking distributions before
totals, and recording decisions before feedback are not stylistic preferences; each one prevents a
specific distortion that would otherwise be invisible.

A small panel can compress the design steps into a single two-hour session, and it cannot compress
the operation steps without losing the measurement properties that make the panel worth having.

The most commonly skipped step is the distribution check, because it feels like statistics rather
than judgement. It is the step that catches leniency, severity, and halo before they reach the
ranking, and skipping it is why panels argue about individual scores instead of about patterns.

A panel with limited time should compress the design steps, not the operation steps. The design can
be agreed in one session; the operation is where the measurement properties are won or lost.

### Decision table: what to do when judges disagree

| Pattern | Likely cause | Action | Record |
|---|---|---|---|
| One team, one criterion, spread of 2+ | Ambiguous anchor | Discuss evidence, then rescore | Reason for change |
| One judge low on everything | Severity | Check anchors; normalise if consistent | Judge mean |
| One judge high on everything | Leniency | Check anchors; normalise if consistent | Judge mean |
| All judges identical on all teams | Halo or central tendency | Check variance; review anchors | Panel SD |
| Scores correlate with presentation order | Order effect | Rescore in a different order | Order and scores |
| One criterion identical for all teams | Non-discriminating | Remove or rewrite next cycle | Criterion |
| Panel split on a knockout finding | Definition ambiguity | Apply the written definition; escalate if needed | Finding and rule |

### Edge cases and forced decisions

**A judge knows a team.** Declare the conflict before scoring. The options are recusal, scoring
without participating in the discussion, or scoring with the conflict recorded and the score
reviewed by a second panel member. The choice is made by the chair, not by the judge.

**A team's data are identifiable in the dossier.** Apply the schedule, record the finding, and
notify the programme lead. This is not a scoring discussion; it is an incident.

**A judge has not read the dossier.** The judge cannot score. Either the session is rescheduled or
the judge is replaced; scoring from the presentation alone would weight the wrong criteria.

**The panel cannot agree on a knockout.** Apply the written definition literally. If the definition
does not resolve it, the panel records the ambiguity and escalates to the programme lead; it does
not invent a new rule for one team.

**A team appeals.** The appeal is against a criterion score or a finding, and it is assessed against
the anchor and the evidence. An appeal that says "we worked hard" is not an appeal.

**The top two teams are within noise.** Report a tie and use the qualitative criteria to make a
recommendation. A ranking that depends on a 0.05 difference in a score with a 0.4 range is not a
ranking.

## 6. Worked example

Three judges score four capstone teams on nine criteria: decision 0.15, data 0.10, validation 0.15,
modelling 0.15, equity 0.10, governance 0.10, system 0.10, value 0.10, reproducibility 0.05.

### The raw scores

Each judge scores independently on a one-to-five scale.

Judge 1 totals: T1 3.60, T2 3.00, T3 4.20, T4 2.25.
Judge 2 totals: T1 4.30, T2 3.30, T3 4.60, T4 2.50.
Judge 3 totals: T1 3.25, T2 3.10, T3 4.00, T4 2.00.

The panel means are T1 3.72, T2 3.13, T3 4.27, T4 2.25.
The ranges are T1 1.05, T2 0.30, T3 0.60, T4 0.50.

The ranking is stable across all three judges: T3 first, T1 second, T2 third, T4 fourth. That is a
good sign, and it does not mean the scores are reliable; it means the ordering is robust to the
panel's disagreement.

### Diagnostic one: judge severity

Judge 2's mean across the four teams is 3.67, against 3.26 for Judge 1 and 3.09 for Judge 3. Judge
2 is 0.58 above Judge 3, which is more than half a scale point.

That is the first diagnostic: a systematic offset. It matters when teams are close, because a
lenient judge can lift a team above another on the strength of the judge's disposition rather than
the work.

The panel normalises: each judge's scores are converted to z-scores within that judge's set, and the
team's normalised score is the mean z across judges. T1 is +0.48, T2 -0.27, T3 +1.23, T4 -1.45.
The ranking is unchanged, which tells the panel that the severity offset was not driving the order.

Normalisation is not a cure. It assumes the judge is consistent and merely offset. If the lenient
judge is also inconsistent, normalisation can make the ranking worse by pretending the scores are
comparable.

### Diagnostic two: the spread on T1

T1's range is 1.05 scale points, from 3.25 to 4.30. The panel's other ranges are 0.30 to 0.60.
That is the second diagnostic: a disagreement large enough to change the team's position if the
other teams were closer.

The panel reads the notes. Judge 2 scored validation 5, citing the calibration plot; Judge 3 scored
validation 3, noting that the calibration slope was not reported and the plot alone does not show
it. The anchor for validation says "reports calibration slope or intercept", so Judge 3's reading is
the correct one and Judge 2 applied the wrong anchor.

The score is revised to 4 for Judge 2, with the reason recorded. T1's total moves to 3.60, 4.00,
3.25, and the mean to 3.62. The ranking is unchanged, and the rubric's anchor is now clearer for
the next cycle.

### Applying a deduction

T2's dossier contains a figure with a date of birth visible in a screenshot. The schedule assigns a
20-point deduction for identifiable protected health information (PHI) in any artefact.

The panel applies the deduction, records the finding, and notifies the programme lead. T2's score
falls from 3.13 to a failing level, and the team is offered a correction window to replace the
figure. The deduction is not a judgement about the team's effort; it is a fixed consequence of a
defined finding.

The arithmetic: a total of 3.13 on a five-point scale is 62.6 out of 100. A 20-point deduction
leaves 42.6, below the pass threshold of 50.

### Applying the individual multiplier

One member of T4 cannot explain the regulatory memo or the validation split, and the contribution
log shows 40 percent of the median hours for three consecutive weeks. The panel assigns a
multiplier of 0.6, with written justification and a second reviewer.

The arithmetic: a team score of 80 with a multiplier of 0.6 is 48. That is deliberately harsh, and
it is the mechanism that protects the credential. The member is offered a remediation route: a
supplementary defense on the two artefacts, with the multiplier reviewed if the defense passes.

### Varying one assumption

Suppose Judge 3 recuses because of a conflict with T3. The panel has two judges, and the weights
must be renormalised across the remaining scores rather than silently dropping a judge.

With two judges, the panel mean for T3 becomes 4.25 and the spread falls, which makes T3 look more
certain than it is. The panel records that the score is based on two judges and that the third was
recused, and it does not report a confidence interval it cannot compute.

The alternative, replacing the judge with someone who has not read the dossiers, is worse: a
replacement scoring from the presentation alone would weight the wrong criteria. Recusal plus a
recorded note is the honest option.

### The panel meeting

The meeting ran to ninety minutes for four teams, which is typical when the outliers are few.

The first twenty minutes were distribution checks: each judge's mean and standard deviation, and
the spread per team. The panel noted Judge 2's leniency and T1's range, and agreed to discuss only
the outliers.

The next forty minutes were adjudication. Three scores changed, each for a stated reason, and each
change was smaller than one level. The panel did not discuss the teams whose scores were within
half a level, which is what kept the meeting short.

The final thirty minutes were decisions: totals, deductions, multipliers, and the wording of the
feedback. The panel wrote one strength and one gap per team, and assigned the feedback to the chair
rather than to individual judges, so that no comment could be attributed.

### What the panel wrote

The T1 record read: "Strong decision framing and a credible comparator; validation is well designed
but the calibration slope is missing from the report, which cost a level on that criterion. The
team should add the slope and a reliability plot before the defense."

The T2 record read: "Good data provenance and a clear analysis plan; the dossier contains a figure
with a date of birth visible, which triggers the 20-point deduction and the incident process. The
team has a correction window before the deduction is final."

The T3 record read: "The most complete validation in the cohort, with temporal and site-held-out
results and a subgroup table. The value case is thin, and the monitoring plan does not name a
trigger. Neither affects the ranking."

The T4 record read: "The decision is real and the data are adequate; the pipeline is not
reproducible and the individual defense revealed a gap in two artefacts. The multiplier applies."

### The tie that was not a tie

T1's raw mean of 3.72 and T2's 3.13 look far apart, and the panel might have stopped there. The
normalised scores, +0.48 and -0.27, are also separated. The panel checked whether the difference
exceeded the spread implied by the judges, and it did: the smallest difference between adjacent
teams in the normalised ranking was 0.75 z-units, well outside the variation attributable to a
single judge's offset.

The check matters more in a close cohort. In a year where the top two teams differ by 0.05 on the
raw scale while the judges' range is 0.4, the honest output is a tie and a qualitative
recommendation, not a ranking.

### What the exercise demonstrates

Three things.

A ranking can be stable and still contain an error: the panel's ordering did not change, and Judge
2's validation score was still wrong against the anchor, which changed T1's mean by 0.1.

Severity is visible in a table and invisible in a discussion. Nobody noticed Judge 2's leniency
until the means were computed, and after they were computed, nobody needed to argue about it.

Deductions are administrative. The panel spent four minutes on the PHI finding, because the
schedule and the process were defined in advance. The alternative, a debate about how serious the
image was, would have taken thirty minutes and produced an inconsistent outcome.

### What the panel would change

The panel's own review produced two changes for the next cycle.

The validation anchor was rewritten to say "reports calibration slope or intercept" rather than
"reports calibration", because the disagreement on T1 came from a word that could be read two ways.
The change is small and it prevents the same argument next year.

The value criterion was flagged as possibly non-discriminating, because all four teams scored 2 or
3 and the scores tracked the presentation rather than the analysis. The panel did not change the
weight immediately; it will check whether the pattern repeats before deciding.

Both changes are small, and both are the kind of change a rubric only gets from a review after the
cycle rather than from a discussion before it.

## 7. Code, query, or calculation

The block computes weighted totals, judge severity, normalised scores, and the effect of a
deduction. It uses only the standard library.

```python
import statistics as st

weights = {"decision": 0.15, "data": 0.10, "validation": 0.15, "modelling": 0.15,
           "equity": 0.10, "governance": 0.10, "system": 0.10, "value": 0.10,
           "repro": 0.05}
teams = ["T1", "T2", "T3", "T4"]
judges = {
    "J1": {"T1": [4, 4, 4, 4, 3, 3, 3, 3, 4], "T2": [3, 3, 3, 3, 3, 3, 3, 3, 3],
           "T3": [5, 4, 5, 4, 4, 4, 3, 4, 4], "T4": [2, 3, 2, 3, 2, 2, 2, 2, 2]},
    "J2": {"T1": [5, 4, 5, 4, 4, 4, 4, 4, 4], "T2": [4, 3, 3, 4, 3, 3, 3, 3, 3],
           "T3": [5, 5, 5, 5, 4, 4, 4, 4, 5], "T4": [3, 3, 2, 3, 2, 2, 2, 3, 2]},
    "J3": {"T1": [3, 4, 3, 4, 3, 3, 3, 3, 3], "T2": [3, 4, 3, 3, 3, 3, 3, 3, 3],
           "T3": [4, 4, 4, 4, 4, 4, 4, 4, 4], "T4": [2, 2, 2, 2, 2, 2, 2, 2, 2]},
}

def total(scores):
    return sum(s * w for s, w in zip(scores, weights.values()))

for judge, rows in judges.items():
    print(judge, {t: round(total(s), 2) for t, s in rows.items()})

print("means and ranges:")
for t in teams:
    vals = [total(judges[j][t]) for j in judges]
    print(f"  {t}: {sum(vals)/len(vals):.2f} (range {min(vals):.2f}-{max(vals):.2f})")

print("judge severity:",
      {j: round(sum(total(s) for s in rows.values()) / len(rows), 2)
       for j, rows in judges.items()})

mu = {j: st.mean([total(s) for s in rows.values()]) for j, rows in judges.items()}
sd = {j: st.pstdev([total(s) for s in rows.values()]) for j, rows in judges.items()}
print("normalised mean z:")
for t in teams:
    z = [(total(judges[j][t]) - mu[j]) / sd[j] for j in judges]
    print(f"  {t}: {sum(z)/len(z):+.2f}")

score_100 = 3.13 / 5 * 100
print(f"T2: {score_100:.1f} on 100; after 20-point deduction {score_100 - 20:.1f}")
print(f"team 80 with multiplier 0.6 = {80 * 0.6:.0f}")
```

Expected output:

```text
J1 {'T1': 3.6, 'T2': 3.0, 'T3': 4.2, 'T4': 2.25}
J2 {'T1': 4.3, 'T2': 3.3, 'T3': 4.6, 'T4': 2.5}
J3 {'T1': 3.25, 'T2': 3.1, 'T3': 4.0, 'T4': 2.0}
means and ranges:
  T1: 3.72 (range 3.25-4.30)
  T2: 3.13 (range 3.00-3.30)
  T3: 4.27 (range 4.00-4.60)
  T4: 2.25 (range 2.00-2.50)
judge severity: {'J1': 3.26, 'J2': 3.67, 'J3': 3.09}
normalised mean z:
  T1: +0.48
  T2: -0.27
  T3: +1.23
  T4: -1.45
T2: 62.6 on 100; after 20-point deduction 42.6
team 80 with multiplier 0.6 = 48
```

The z-scores are computed within each judge's four-team set, which is a small sample; with four
teams, a single extreme score moves the standard deviation substantially. In a real panel with
twenty teams the normalisation is more stable, and the panel should still report the raw scores
alongside the normalised ones.

### What the output does not tell you

The table gives a ranking, a spread, and a severity estimate. It does not give a confidence interval
for the ranking, because four teams and three judges cannot support one.

A panel that wants to know whether the top two are distinguishable should look at the spread of
scores for those teams and at the size of the difference. If the difference is smaller than the
average range across judges, the ranking is not established.

A second thing the output does not tell you is whether the rubric measured the right thing. A
perfectly consistent ranking of the wrong quality is possible, and no amount of arithmetic will
detect it.

### A note on the deduction arithmetic

The deduction is applied to the percentage score, not to the five-point total, and the order
matters. Applying a 20-point deduction to a 3.13 out of 5 gives 3.13 - 4 = negative, which is
meaningless; converting to 62.6 out of 100 and then subtracting gives 42.6, which is a failing
score. The schedule should state the scale on which deductions apply, and the panel should apply it
consistently.

The individual multiplier is applied after deductions, because the multiplier is about the
individual's understanding and the deductions are about the team's artefact. Reversing the order
produces a different and indefensible result.

## 8. Data, standards, and regulatory context

Judging is an assessment practice, and the relevant standards are educational and professional
rather than clinical. Three bodies of guidance apply, and one legal regime constrains what can be
scored.

### Assessment standards

The *Standards for Educational and Psychological Testing*, published jointly by the American
Educational Research Association, the American Psychological Association, and the National Council
on Measurement in Education, set out the properties an assessment should have: validity,
reliability, fairness, and consequences.
The standards are not binding on a course, and they name the questions a well-designed rubric should
answer. Does the rubric measure the construct it claims? Are scores consistent across raters and
occasions? Is the assessment fair to groups with different backgrounds? What are the consequences of
using the scores?

A capstone rubric that omits fairness is vulnerable to the criticism that it rewards presentation
style, English fluency, or institutional background rather than evidence. Naming those risks and
mitigating them is part of the design.

### Research integrity

Fabrication, falsification, and plagiarism are research misconduct, and a capstone is not exempt.
The standard deduction schedule treats a fabricated result as a zero rather than a deduction,
because the finding is about integrity rather than quality.
Institutional policies define the process: a finding is reported to the programme lead, who follows
the institution's misconduct procedure. The panel does not adjudicate misconduct; it records the
finding and refers it.

### Data protection

A dossier or demo that contains identifiable patient data is a data protection incident, not a
scoring issue. Under HIPAA, the covered entity's privacy officer manages the response; under the
GDPR, the controller does. The panel's role is to record the finding, apply the schedule, and notify
the responsible officer.

The distinction matters because panels sometimes treat a PHI finding as a quality problem that can
be offset by strong evidence. It cannot. The finding triggers a separate process.

### Conflicts of interest

Professional bodies, including the International Committee of Medical Journal Editors, require
disclosure of financial and personal relationships that could bias judgement. A capstone panel is
not a journal, and the principle applies: a judge who has supervised a team, employed a member, or
has a financial interest in the work should declare it before scoring.

The management options are recusal, non-participation in the discussion, or scoring with review.
The choice depends on the panel's size and the nature of the relationship, and it is recorded.

### Where the standards conflict

Confidentiality and feedback can conflict: a team may want to know who said what, and the panel's
deliberations are confidential. The resolution is to report the criterion, the score, and the
reason, without attributing the comment to a judge.

Transparency and security can conflict: publishing the rubric helps teams, and publishing the
anchor submissions may expose past teams' work. The resolution is to publish the rubric and
anonymised anchors.

### What a capstone panel is not

It is not a misconduct investigation. It records findings and refers them to the institutional
process.
It is not a legal proceeding. The regulatory memo is a triage, and the panel judges its reasoning,
not its legal correctness.
It is not a peer review. The panel does not decide whether the work is publishable; it decides
whether the evidence supports the claim and whether the team understands it.

Keeping those boundaries clear protects both the panel and the teams. A panel that drifts into
misconduct investigation produces a process that is neither fair nor final.

A second boundary is between the panel and the programme. The panel decides the score and the
decision for each team; the programme decides the rubric, the weights, and the appeals route. A
panel that changes a weight mid-cycle has crossed the boundary, and a programme that overrides a
score without a record has crossed it in the other direction.

| Instrument | Status | What it constrains |
|---|---|---|
| AERA/APA/NCME Standards | Professional convention | Validity, reliability, fairness |
| Institutional misconduct policy | Binding | How findings are handled |
| HIPAA Privacy Rule | Binding law in the US | Identifiable data in artefacts |
| GDPR | Binding law in the EU | Processing of personal data |
| ICMJE disclosure | Professional convention | Conflicts of interest |
| Programme rubric | Contractual | Criteria, weights, deductions |

## 9. Industry practice and case evidence

### What the measurement literature shows about rubrics

The research on rubrics is consistent on one point and divided on another. Rubrics improve
consistency when they are analytic and anchored; they do not eliminate rater effects, and a
holistic score can hide them.

Lai, Wolfe, and Vickers, in a 2015 study in *Educational and Psychological Measurement*, compared
designs in which different raters scored different traits with designs in which one rater scored all
traits. The all-traits design produced clear illusory halo: within-trait variance was 32 percent
smaller, intercategory correlations were inflated (about .67 versus .54), and the effective test
length fell to roughly a third. The study is about writing assessment, not capstones, and the
mechanism transfers: a judge who scores every criterion after forming one impression will produce
correlated scores that look like a profile and are mostly one judgement repeated.

Harsch and Martin, in a 2012 study in *Assessment in Education*, compared holistic and analytic
scoring and found that holistic scores can mask differences in how raters interpret descriptors.
Their recommendation, to combine holistic judgement with descriptor-focused analytic scores to
monitor rating quality, is directly applicable to a capstone panel that wants both a quick
impression and a defensible profile.

Kim's 2020 many-facet Rasch analysis in *Language Testing in Asia* examined how the order of
criteria affects halo. The finding that random presentation slightly reduced halo is a small
practical lever: a scoring form that varies criterion order across judges makes a single impression
slightly harder to carry through every criterion.

### What this means for a capstone panel

Three practices follow from the evidence.

Score analytically, with anchors, even when time is short. A holistic score is faster and less
defensible, and it hides the halo that the panel most needs to see.

Score independently before discussing. The measurement value of three judges comes from three
independent observations; a discussion first collapses them into one.

Check the distribution before the total. Severity, central tendency, and restriction of range are
visible in a table of means and standard deviations, and they are invisible in a ranked list.

### What industry panels do

Funding panels and procurement committees use the same structure: published criteria, independent
scoring, a chair who manages conflicts, and a recorded decision. The differences are in the stakes
and the paperwork, not in the logic.
A data scientist who has served on a capstone panel has practised the mechanics of a grant review:
score independently, justify against the criterion, declare conflicts, and record the decision.

The difference in industry is the consequence of a wrong call. A funding panel that ranks a weak
proposal first wastes money; a procurement committee that approves an unvalidated model creates a
safety risk. The mechanics are worth learning in a setting where the consequence is a grade.

### What the evidence does not settle

The measurement literature says little about capstone panels specifically, and the transfer from
writing assessment to biomedical project review is an inference rather than a finding. The
mechanisms, halo, severity, and order effects, are general features of human judgement, and the
mitigations are general too.

What the literature does not settle is the right number of criteria or the right weight for
evidence. Those are programme decisions, and a programme should make them explicitly rather than
inheriting them from a template.

### A note on judging your own work

The hardest judging problem is the one a team faces before submission: scoring its own dossier.
The halo that affects panels affects teams more, because the team knows the effort behind every
artefact and reads quality into it.

The mitigation is the stranger test from Topic 09. A stranger has no halo to apply, and the
stranger's reading is a better estimate of the panel's than the team's own.

### Inference, labelled

It is an inference that a calibrated panel produces rankings that survive a change of judge. The
mechanism is that calibration aligns anchors and exposes severity, and the evidence from the
measurement literature supports the components. No controlled study of capstone panels is cited.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Rubric written after scoring | Criteria match the winners | Rationalisation | Unfair ranking | Publish before submissions |
| No anchors | Judges argue about words | Ambiguity | Unreliable scores | Behavioural descriptors |
| Discussion before scoring | One strong opinion sets the tone | Anchoring | Consensus impression | Independent forms first |
| Halo | All criteria move together | One impression | Flat profile | Analytic scoring, varied order |
| Leniency or severity | One judge's mean far from panel | Disposition | Distorted close calls | Normalise and record |
| Central tendency | All scores 3 or 4 | Risk aversion | No discrimination | Anchor extremes |
| Order effect | Later teams score lower | Fatigue | Position bias | Rotate order, breaks |
| Affinity bias | Judges favour similar teams | Similarity | Inequity | Blind review where possible |
| Undeclared conflict | Judge supervises a team | Relationship | Invalid score | Declare before scoring |
| Deduction invented on the day | Penalty not in the schedule | Ad hoc judgement | Inconsistent treatment | Fixed schedule |
| PHI treated as quality issue | Panel debates the score | Category error | Incident mishandled | Refer to privacy officer |
| Multiplier without evidence | One judge's impression | Unaccountable | Appeal and grievance | Written justification, second reviewer |
| No feedback | Teams learn only the rank | Withholding | No improvement | Criterion-level notes |
| Rubric never revised | Same criteria for years | Inertia | Non-discriminating criteria persist | Post-cycle review |

The table is ordered from design failures to process failures. Design failures affect every team;
process failures affect the teams whose scores happen to be close.

Two rows deserve emphasis because they are invisible in the final ranking. Halo produces a flat
profile that looks like a consistent team, and central tendency produces scores that look fair. Both
are detectable only in the distribution, which is why the distribution check is not optional.

The mitigations are procedural rather than motivational. "Be objective" is not a mitigation;
"collect forms before the meeting" is.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Criteria measure the claim | Validity depends on it | Criteria track chapters | Map each criterion to the evidence bar |
| Anchors are observable | Reliability depends on them | Abstract descriptors | Test on an anchor case |
| Judges score independently | Measurement depends on it | Pre-meeting discussion | Collect forms before the meeting |
| Weights reflect priorities | Fairness depends on them | Weights set by habit | Publish and review |
| Normalisation corrects severity | Close calls depend on it | Inconsistent judge | Check judge SD |
| Deductions are pre-defined | Consistency depends on it | Ad hoc penalties | Schedule and record |
| Multipliers are evidenced | Fairness depends on it | Impression only | Contribution log and defense |
| Panels have time to calibrate | Quality depends on it | Schedule pressure | Ninety minutes in the agenda |
| Anchor cases represent the range | Calibration transfers to live work | All anchors are strong submissions | Include one weak and one middling case |

The largest limit is that a rubric measures what it measures. A rubric weighted toward evidence will
not identify the team with the best product sense, and a rubric weighted toward adoption will not
identify the most rigorous work. The panel should know which quality it is choosing and say so.

A second limit is that the arithmetic is not the decision. A panel that follows the numbers off a
cliff has confused measurement with judgement. The numbers inform the decision; the panel records
the reasoning.

A third limit is that calibration takes time the panel may not have. A ninety-minute session before
the event is the single highest-value investment in reliability, and a panel that skips it will
spend longer arguing during the event and produce a less defensible result.

A fourth limit is that some qualities resist scoring. Whether a project is important, or whether the
team showed unusual judgement under pressure, are judgements the rubric can capture only indirectly.
The panel should say so in the feedback rather than invent a criterion for them.

## 12. Exercises

### Exercise 1. Write anchors for one criterion (application, 40 minutes)

Choose the validation criterion and write five level descriptors from inadequate to exemplary, each
describing observable features of a report.
**Deliverable:** five descriptors.
**Criteria:** each descriptor names something a reader can see; no descriptor uses "good" or
"poor"; the levels are distinguishable.
**Time:** 40 minutes.

### Exercise 2. Score three anchor submissions (application, 60 minutes)

Score three short submissions independently, then compare with a partner and discuss the
disagreements.
**Deliverable:** three score sheets and a note on the largest disagreement.
**Criteria:** independent scoring before discussion; disagreement tied to a descriptor; a proposed
fix to the anchor.
**Time:** 60 minutes.

### Exercise 3. Compute severity and normalise (calculation, 30 minutes)

Using the score matrix in Section 7, compute each judge's mean and the normalised z-scores, and
state whether the ranking changes.
**Deliverable:** the table and a one-paragraph interpretation.
**Criteria:** correct arithmetic; the limits of normalisation stated; the ranking comparison made.
**Time:** 30 minutes.

### Exercise 4. Diagnose a score set (critique, 35 minutes)

Given a matrix where all judges give every team a 4 on one criterion and one judge is 0.8 below the
panel on all teams, identify the failures and propose two fixes.
**Deliverable:** a diagnosis and a fix list.
**Criteria:** names central tendency and severity; fixes address the rubric and the process; no fix
is "tell judges to be more careful".
**Time:** 35 minutes.

### Exercise 5. Design the deduction schedule (design, 30 minutes)

Write a deduction schedule for six findings, from fabricated results to missing subgroup analysis,
with points and a rationale for each.
**Deliverable:** a six-row table.
**Criteria:** each finding is defined observably; points are proportionate; the most serious
findings trigger a process rather than a deduction.
**Time:** 30 minutes.

### Exercise 6. Run a calibration session (application, 60 minutes)

Run a ninety-minute session with three judges on two anchor cases, including independent scoring,
discussion, and rescoring.
**Deliverable:** a session record with before-and-after scores.
**Criteria:** independence preserved; disagreements recorded; at least one anchor revised.
**Time:** 60 minutes.

### Exercise 7. Write feedback for a team (application, 30 minutes)

Write criterion-level feedback for a team that scored 2 on validation, 3 on governance, and 4 on
decision, with one action per gap.
**Deliverable:** a 250-word note.
**Criteria:** every claim tied to a criterion; one action per gap; no reference to a judge.
**Time:** 30 minutes.

### Exercise 8. Audit a rubric (critique, 40 minutes)

Take the rubric in Section 7 and identify one criterion that may not discriminate and one weight
that may reward the wrong behaviour. Propose a revision.
**Deliverable:** a 300-word audit and a revised row.
**Criteria:** the critique is about measurement, not about the teams; the revision is testable; the
consequence is stated.
**Time:** 40 minutes.

**Assessment note.** Exercises 1, 3, and 5 have checkable outputs. Exercises 2 and 6 are assessed on
the process record: whether scoring was independent, whether disagreements were tied to descriptors,
and whether the anchor changed.

Exercise 2 is the one that changes a judge's confidence. Most people discover that their scores
differ from a colleague's by a full level on at least one criterion, and that the difference comes
from the anchor rather than the work.

Exercise 6 is the one that changes a panel's behaviour. A panel that has calibrated once will ask
for anchors the next time, because the difference between arguing about a score and checking a
descriptor is immediately obvious.

Exercise 8 is the one a programme should run annually. A rubric accumulates criteria that no longer
discriminate, and the annual audit is what keeps it an instrument rather than a ritual.

## 13. Self-check answers

**1. Why is reliability not validity?**
Reliability is consistency; validity is whether the instrument measures the intended construct. A
panel can agree consistently on presentation quality while claiming to measure evidence quality,
which is reliable and invalid.

**2. What does an anchor do?**
It describes observable performance at a level, so that two judges reading the same artefact apply
the same standard. Without anchors, the criterion is a word whose meaning varies by judge.

**3. Why must judges score before discussing?**
Because discussion produces anchoring, and the value of multiple judges is independent observation.
Once one judge states a view, the others' scores partly measure that view rather than the work.

**4. Compute Judge 2's severity relative to the panel.**
Judge 2's mean is 3.67; the panel mean is (3.26 + 3.67 + 3.09) / 3 = 3.34. Judge 2 is 0.33 above
the panel, which is a leniency offset of about a third of a scale point.

**5. What does normalisation correct, and what does it not?**
It corrects a consistent offset in a judge's scale. It does not correct inconsistency, bias, halo,
or scoring the wrong construct.

**6. Why is a PHI finding not a scoring issue?**
Because it triggers a separate legal and institutional process. Treating it as a quality problem
that strong evidence can offset would leave the incident unmanaged.

**7. What is the smallest defensible response to a judge with a conflict?**
Declare it before scoring and let the chair decide between recusal, non-participation in discussion,
and reviewed scoring. The judge does not decide their own management.

**8. When should a panel report a tie?**
When the difference between two teams is smaller than the spread the panel's own disagreement
implies. A ranking that depends on noise is not a ranking.

**9. What makes feedback actionable?**
It names the criterion, the observed feature, and one action. "Validation was weak" is not
actionable; "the report gives an interval but no calibration plot; add a reliability plot and a
slope" is.

**10. Rubric for Exercise 5.** A strong schedule defines each finding observably, assigns
proportionate points, and routes integrity and privacy findings to a separate process. A weak
schedule lists vague findings and uniform penalties, which makes application inconsistent.

**11. Rubric for Exercise 7.** A strong note names the criterion, the observed feature, and one
action, and it does not mention a judge or a comparison with another team. A weak note offers
encouragement and a general suggestion to improve validation.

**12. What is the difference between a deduction and a multiplier?**
A deduction is about the artefact and applies to the team's score; a multiplier is about an
individual's demonstrated understanding and applies to that member's score after the team score is
settled.

**13. Why is moderation across panels necessary?**
Because two panels applying the same rubric can produce different distributions, and without
comparison the programme cannot tell whether the difference is the work or the standard.

**14. What does a non-discriminating criterion look like in a score set?**
Every team receives the same score, or scores vary by less than half a level. The criterion is not
separating teams, and the panel should ask whether it is measuring anything the other criteria do
not.

**15. How should a panel treat a team that improved a lot during the project?**
The rubric scores the artefact and the defense, not the trajectory. Improvement belongs in the
feedback, not in a score, unless the programme has a criterion for it and has published it.

## 14. Interview questions

**Q1. "Have you judged work? How did you ensure consistency?"**
Strong answer describes anchors, independent scoring, and calibration. Weak answer says the team
agreed on a winner.

**Q2. "What is halo, and how do you detect it?"**
Strong answer defines it and names the correlation across criteria as the signal. Weak answer says
it is being biased by one good feature.

**Q3. "How would you handle a judge who is much harsher than the others?"**
Strong answer checks anchors, then normalises with a record, and distinguishes severity from
inconsistency. Weak answer says replace the judge.

**Q4. "Why publish the rubric?"**
Strong answer says it lets teams allocate effort and makes the panel accountable. Weak answer says
it is required.

**Q5. "What do you do with a team whose scores are all identical across judges?"**
Strong answer suspects halo or a non-discriminating criterion and reviews the variance. Weak answer
says the team was clearly average.

**Q6. "How do you weight evidence against presentation?"**
Strong answer gives weights and a rationale tied to the decision. Weak answer says both matter
equally.

**Q7. "What is a knock-out criterion?"**
Strong answer describes a finding that ends the assessment regardless of other scores. Weak answer
calls it a low score.

**Q8. "How do you keep an individual multiplier fair?"**
Strong answer requires evidence, a bounded range, written justification, and review. Weak answer
says the panel decides.

**Q9. "What would you change about a rubric you have used?"**
Strong answer names a criterion that did not discriminate or a weight that rewarded the wrong
behaviour. Weak answer says nothing.

**Q10. "How do you give feedback that a team can use?"**
Strong answer ties it to criteria and actions. Weak answer says be encouraging.

**Q11. "What would you do if you disagreed with the panel's ranking?"**
Strong answer describes recording the disagreement with its evidence, applying the adjudication
rule, and escalating if the rule does not resolve it. Weak answer says accept the majority.

A strong answer to this set treats the rubric as a shared instrument rather than a personal
standard. The candidate who says "the anchor says this, and the report does not meet it" is easier
to work with than the candidate who says "it felt like a 3".

## 15. Key terms

- **Adjudication** — resolving score disagreement by rule.
- **Analytic rubric** — separate scores per criterion.
- **Anchor** — observable description of a level.
- **Affinity bias** — favouring similar teams.
- **Calibration** — aligning judges before scoring.
- **Central tendency** — avoiding extremes.
- **Conflict of interest** — relationship that could bias judgement.
- **Criterion** — dimension of performance.
- **Deduction** — fixed penalty for a finding.
- **Halo effect** — one impression contaminating criteria.
- **Hindsight bias** — judging process by outcome.
- **Holistic rubric** — single overall score.
- **Inter-rater reliability** — consistency across judges.
- **Leniency** — systematic high scoring.
- **Multiplier** — factor applied to an individual score.
- **Normalisation** — adjusting for severity.
- **Order effect** — position influencing scores.
- **Restriction of range** — scores clustered narrowly.
- **Severity** — systematic low scoring.
- **Weight** — share of total per criterion.
- **Z-score** — score in standard deviations.
- **Moderation** — comparing distributions across panels.
- **Pass threshold** — score at which the decision changes.
- **Score sheet** — form on which a judge records scores.
- **Criterion-level feedback** — comments tied to a named criterion.

The list is short because the concepts are few and the practice is in the application. A judge who
knows these terms and has run one calibration session is better prepared than one who has read a
longer glossary and scored alone.

## 16. Sources

All URLs accessed 26 October 2026.

- American Educational Research Association, American Psychological Association, National Council on
  Measurement in Education. *Standards for Educational and Psychological Testing*, 2014.
  https://www.testingstandards.net/
- Lai ER, Wolfe EW, Vickers D. Differentiation of illusory and true halo in writing scores.
  *Educational and Psychological Measurement* 2015;75(1):102-125.
  https://doi.org/10.1177/0013164414527449
- Harsch C, Martin G. Comparing holistic and analytic scoring methods. *Assessment in Education*
  2012;19(3):281-307. https://doi.org/10.1080/0969594X.2012.742422
- Kim HJ. Effects of rating criteria order on the halo effect. *Language Testing in Asia*
  2020;10:18. https://doi.org/10.1186/s40468-020-00115-0
- ICMJE, *Disclosure of financial and non-financial relationships and activities*.
  https://www.icmje.org/recommendations/browse/roles-and-responsibilities/author-responsibilities--conflicts-of-interest.html
- HIPAA Privacy Rule, 45 CFR Parts 160 and 164. https://www.hhs.gov/hipaa/
- Regulation (EU) 2016/679, General Data Protection Regulation.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378
- Vasey B, Nagendran M, Campbell B, et al. DECIDE-AI. *Nature Medicine* 2022;28:924-933.
  https://doi.org/10.1038/s41591-022-01772-9

Source note: the measurement literature cited here comes from educational assessment, and its
transfer to capstone panels is an inference. The standards are professional conventions; the
institutional and legal instruments are binding within their scope.

## 17. Links

The rubric in this topic is applied to the deliverables in Topic 09 and to the pitch in Topic 11. A
team that reads all three knows what the panel is measuring before it builds anything.

A note on sequence: read Topic 09 for the artefacts, this topic for how they are scored, and Topic
11 for how the decision is presented. The three are one system, and the weights connect them.

The single most useful sentence in this topic, if a reader remembers only one: score independently,
then talk.

A second candidate for that sentence: publish the rubric before the work, and revise it after the
cycle. Both are cheap, and both change what teams build.

A third: write the decision down. A panel that cannot reproduce its own reasoning a month later has
not made a decision; it has held a meeting.

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_deliverables.md`](09_deliverables.md)
- Next topic: [`11_demo-day.md`](11_demo-day.md)
- Related topic: [`03_team-roles.md`](03_team-roles.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
