# Ethics and Professional Conduct

> Module 00 — Orientation and Industry Map
> Topic 08 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 05, 07
> Version 2026.3

## 1. Why this topic exists

Ethics in biomedical data science is usually taught as a list of principles: autonomy,
beneficence, non-maleficence, justice. The principles are correct and almost useless in the
moment they are needed. The moment is specific: a result is inconvenient, a deadline is close, a
commercial partner wants a stronger claim, a subgroup performs worse, or a dataset was obtained
under an agreement the team has quietly outgrown.

In those moments, the question is not "which principle applies". It is "what do I do next, who do
I tell, and what do I write down". This topic answers that question. It covers the professional
obligations that attach to data work in a regulated industry, the conflicts that arise in
commercial settings, the handling of inconvenient results, authorship and attribution, escalation,
and the failures that recur in this field.

The framing is practical. Ethics is not a separate review that happens at the end. It is a set of
operating habits that determine whether the work can be defended five years later, in front of a
regulator, a patient, or your own colleagues.

### 1.1 The four pressures

Most ethical failures in this field are not caused by bad people. They are caused by ordinary
people responding to four pressures at once.

| Pressure | Typical form | Effect on judgement |
|---|---|---|
| Commercial | "The investor deck needs a number by Friday" | Overstates effect, hides uncertainty |
| Career | "This result would make the project" | Selects the analysis that confirms |
| Institutional | "The chair wants this to work" | Avoids escalating upward |
| Time | "We can fix the subgroup later" | Defers a known harm |

Each pressure is legitimate in isolation. Combined, they produce the quiet decisions that become
retractions, regulatory findings, or harm.

### 1.2 Why professional conduct is a technical skill

A data scientist who cannot document a decision, disclose a limitation, or escalate a concern is
not fully competent, regardless of modelling skill. The technical work is only defensible if the
record shows what was known, when, and what was done about it.

This is also a market reality. Employers in regulated settings hire for judgement about exactly
these situations. An interview question about an inconvenient result is testing whether you have
a process or only an intention.

### 1.3 Scope and limits

This topic is not legal advice and not a substitute for an ethics committee, an institutional
review board (IRB), or counsel. It covers professional expectations, not statutory duties in every
jurisdiction. Where a law is named, it is named for orientation.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Identify the four pressures and the point at which they distort a result | Exercise 1; self-check 2 |
| 2 | Distinguish research misconduct from questionable research practice | Exercise 2; self-check 3 |
| 3 | Apply a twelve-step procedure to a live concern | Exercise 3; interview question 5 |
| 4 | Compute and interpret a subgroup performance gap before deciding to ship | Exercise 4; self-check 5 |
| 5 | Apply authorship criteria and resolve a contribution dispute | Exercise 5 |
| 6 | Write a disclosure statement for a model with a known limitation | Exercise 6 |
| 7 | Recognize a conflict of interest and choose a management strategy | Exercise 7 |
| 8 | Design a project so that inconvenient results are reportable by default | Exercise 8 |

## 3. The core mental model

Ethics operates at three concentric levels: what is legal, what is professionally expected, and
what you can defend personally. The three overlap but do not coincide. A decision can be legal and
professionally questionable, or professionally acceptable and personally corrosive.

### 3.1 Three rings

```text
        +-------------------------------------------+
        |  LEGAL                                    |
        |  statutes, regulations, contracts         |
        |   +-----------------------------------+   |
        |   |  PROFESSIONAL                     |   |
        |   |  codes, standards, norms,         |   |
        |   |  institutional policy             |   |
        |   |   +---------------------------+   |   |
        |   |   |  PERSONAL                 |   |   |
        |   |   |  what you can defend      |   |   |
        |   |   |  in public, later         |   |   |
        |   |   +---------------------------+   |   |
        |   +-----------------------------------+   |
        +-------------------------------------------+
```

The outer ring is the minimum. The inner ring is what keeps you in the profession. Most scandals
occur when someone stays inside the legal ring and ignores the other two.

### 3.2 The decision ladder

When a concern appears, move up the ladder only as far as necessary, but do not stop early.

```text
  1. Notice        Something does not fit: a number, a claim, a data flow.
        |
  2. Name          Say it explicitly: "the subgroup sensitivity is lower".
        |
  3. Check         Is there a rule, a contract, or a policy that applies?
        |
  4. Consult       Ask someone who knows: methodologist, privacy, counsel.
        |
  5. Document      Write the facts, the options, and the decision.
        |
  6. Escalate      Move to the level with authority to decide.
        |
  7. Act           Implement the decision, including the uncomfortable parts.
        |
  8. Disclose      Tell the people affected, in writing, in plain language.
```

The ladder is sequential for a reason. Most failures skip from step 1 to step 7: someone notices,
decides privately, and acts without documenting or consulting.

### 3.3 The asymmetry of harm

In clinical data work, the cost of a false negative (missing a sick patient) and a false positive
(alerting on a well one) are borne by different people. The model developer bears neither. That
asymmetry is the structural reason ethics needs explicit attention.

| Error | Who bears the cost | How visible is it |
|---|---|---|
| False negative | Patient, family | Sometimes, after harm |
| False positive | Patient, clinician, system | As workload and anxiety |
| Biased subgroup performance | Members of the subgroup | Rarely, unless stratified |
| Privacy breach | Patients, institution | Sometimes, years later |
| Overstated claim | Buyer, patients, profession | Eventually, in review |

The developer sees aggregate metrics. The harm is distributed and delayed. A good practice is to
name the harm bearer for each error type before the model is built.

### 3.4 The documentation principle

If it is not written down, it did not happen. In a regulated environment, the record is the
evidence of professional conduct. A decision that was reasonable but undocumented is
indistinguishable, after the fact, from a decision that was careless.

The practical rule: every material decision gets a short note with date, options considered,
rationale, and owner. The note does not need to be long. It needs to exist.

### 3.5 The four questions before any consequential decision

1. Who is affected if this is wrong?
2. What would I need to disclose for this to be honest?
3. Would I make the same choice if the affected person were in the room?
4. What will the record show in three years?

The questions are not a complete framework. They are fast enough to use in a meeting, which is
where most decisions are actually made.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Ethics | Normative study of right conduct | Confused with law | Discipline |
| Professional conduct | Behaviour expected of a member of a profession | Confused with etiquette | Standard |
| Research misconduct | Fabrication, falsification, or plagiarism in research | Applied to every error | Regulatory category |
| Fabrication | Inventing data or results | Confused with imputation | Misconduct |
| Falsification | Manipulating or omitting data to misrepresent | Confused with cleaning | Misconduct |
| Plagiarism | Using others' work without attribution | Confused with paraphrase | Misconduct |
| Questionable research practice (QRP) | Practices that distort the literature without meeting misconduct definitions | Treated as misconduct or as trivial | Category |
| HARKing | Hypothesizing after results are known | Confused with exploratory analysis | QRP |
| p-hacking | Selective analysis to obtain significance | Confused with legitimate sensitivity analysis | QRP |
| Outcome switching | Changing the primary outcome after the study | Confused with protocol amendment | QRP |
| Publication bias | Selective publication of positive results | Confused with editorial preference | Bias |
| Conflict of interest (COI) | Personal interest that could bias professional judgement | Assumed to require financial interest | Governance |
| Financial COI | COI involving money, equity, or fees | Confused with all COI | Governance |
| Non-financial COI | COI involving career, ideology, relationships | Often omitted from disclosures | Governance |
| Undue influence | Pressure that compromises independent judgement | Confused with strong advocacy | Governance |
| Authorship | Attribution for substantive intellectual contribution | Assigned by seniority | Credit |
| Gift authorship | Authorship without qualifying contribution | Treated as courtesy | Misconduct-adjacent |
| Ghost authorship | Omitted author who contributed | Confused with editorial assistance | Misconduct-adjacent |
| Contributorship | Specification of who did what | Confused with authorship order | Credit |
| Informed consent | Voluntary agreement based on adequate information | Confused with a signature | Legal and ethical standard |
| Secondary use | Use of data for a purpose other than the original | Assumed to be covered by consent | Data governance |
| Minimum necessary | Limit data to what is needed for the purpose | Applied only to humans | HIPAA standard |
| De-identification | Removal of identifiers under a defined standard | Assumed to eliminate risk | Legal standard |
| Re-identification | Recovery of identity from de-identified data | Assumed to be impossible | Privacy harm |
| Data provenance | Record of origin and transformation | Assumed to be the EHR | Metadata |
| Algorithmic bias | Systematic error that disadvantages a group | Confused with statistical bias | Harm category |
| Representational harm | Stereotyping or erasure in data or outputs | Confused with allocative harm | Harm category |
| Allocative harm | Unequal distribution of resources or attention | Confused with accuracy | Harm category |
| Privacy harm | Loss of control over personal information | Confused with breach notification | Harm category |
| Duty to warn | Obligation to inform a person at risk | Confused with clinical duty | Legal and ethical concept |
| Whistleblowing | Reporting wrongdoing outside the chain of command | Confused with complaint | Protected activity |
| Escalation | Raising a concern to a level with authority | Confused with going over someone's head | Procedure |
| Retraction | Withdrawal of a published work | Confused with correction | Publishing action |
| Correction | Fixing an error in published work | Confused with retraction | Publishing action |
| Expression of concern | Editorial notice about unresolved questions | Confused with retraction | Publishing action |
| ORI | US Office of Research Integrity | Confused with an IRB | Federal office |
| COPE | Committee on Publication Ethics | Confused with a regulator | Membership body |
| ICMJE | International Committee of Medical Journal Editors | Confused with a funder | Standards body |
| Declaration of Helsinki | Ethical principles for medical research involving humans | Confused with law | Declaration |
| Belmont Report | US principles for human-subjects research | Confused with regulation | Report |
| IRB | Institutional review board | Confused with an ethics committee in other countries | Governance body |
| DSMB | Data and safety monitoring board | Confused with an IRB | Oversight body |
| Sunshine Act | US physician payment transparency law | Confused with a COI policy | Statute |
| GDPR Article 22 | Right not to be subject to solely automated decisions with significant effects | Assumed to apply to all models | Regulation |
| Model card | Structured model documentation | Treated as a disclosure | Documentation |
| Datasheet | Structured dataset documentation | Treated as a license | Documentation |
| Duty to document | Professional obligation to record decisions | Confused with bureaucracy | Practice |

Four definitions deserve expansion.

**Research misconduct.** In the US, the federal definition covers fabrication, falsification, and
plagiarism in proposing, performing, or reviewing research. It does not cover honest error or
differences of opinion. Many serious problems in data science, such as p-hacking or outcome
switching, are not misconduct under this definition but are still professionally unacceptable.

**Conflict of interest.** A COI is a situation, not a behaviour. Having a COI is not wrongdoing.
Failing to disclose and manage it is. Non-financial conflicts, such as a career interest in a
particular result, are as important as financial ones and are less often disclosed.

**Secondary use.** Data collected for care or for one study may be used for another purpose only
if the legal basis and the consent arrangements permit it. "It was already in the warehouse" is
not a legal basis.

**Duty to document.** The obligation to record decisions is not administrative. It is what allows
a colleague, an auditor, or a court to reconstruct why a choice was made. In regulated work, the
record is part of the product.

## 5. The mechanics, step by step

This section is a twelve-step procedure for handling an ethical concern. It is written for the
person who notices the problem, not for the committee that later reviews it.

### Step 1: Notice and name the concern

**Input:** a result, a request, or a data flow that does not fit.
**Output:** one sentence that states the concern without accusation.

**Reasoning.** Concerns are often vague at first: "this feels off". Naming the specific fact
converts a feeling into something that can be examined. "The Spanish-language subgroup has 14
percentage points lower sensitivity" is examinable. "The model seems biased" is not.

**Branch point.** If the concern is about a person's conduct rather than a result, name the
behaviour, not the motive. Motive is rarely knowable and usually counterproductive to assert.

### Step 2: Separate fact from inference

**Input:** the concern statement.
**Output:** two columns: what is observed and what is inferred.

**Reasoning.** The distinction determines what can be said publicly. "Sensitivity is 0.58 in the
Spanish subgroup" is a fact from the analysis. "The team does not care about equity" is an
inference and probably wrong. Keeping the columns separate protects the credibility of the
concern.

**Branch point.** If the fact depends on a calculation, reproduce it before raising it. A concern
based on a misread table damages trust in the next concern.

### Step 3: Identify who is affected

**Input:** facts.
**Output:** a list of affected parties and the type of harm.

**Reasoning.** Ethical analysis requires naming who bears the cost. Patients in the subgroup,
clinicians who act on the output, the institution, and the payer may all be affected differently.

**Branch point.** If no one is affected, the issue may be a preference rather than an ethical
concern. That is still worth discussing, but it should not be escalated as harm.

### Step 4: Check the applicable rules

**Input:** facts and affected parties.
**Output:** a list of applicable laws, contracts, policies, and professional norms.

**Reasoning.** Rules narrow the space of acceptable actions and sometimes require disclosure. A
data use agreement may prohibit the secondary analysis; an IRB protocol may require reporting a
protocol deviation; a contract may require notification of a security incident.

**Branch point.** If a rule requires disclosure, the decision is no longer discretionary. Follow
the required process and document it.

### Step 5: Consult before deciding

**Input:** facts, affected parties, rules.
**Output:** at least one external perspective.

**Reasoning.** The person closest to the work is the least able to see it clearly. A consultation
with a methodologist, a privacy officer, or counsel tests the framing. It also creates a record
that the concern was raised.

**Branch point.** If the only available consultee is the person causing the pressure, go to the
next level. Consultation with the source of the pressure is not consultation.

### Step 6: Generate at least three options

**Input:** consultation.
**Output:** three options, including one that is uncomfortable.

**Reasoning.** Binary framing ("ship or delay") produces bad decisions. Real options often
include: ship with a disclosed limitation and monitoring; narrow the population; add a mitigation;
run a targeted validation; escalate for a decision. The uncomfortable option is included so that
it is considered rather than avoided.

**Branch point.** If all options protect the project and none protects the affected group, the
framing is wrong. Return to step 3.

### Step 7: Assess each option against the three rings

**Input:** options.
**Output:** a table of legality, professional acceptability, and personal defensibility.

**Reasoning.** An option that is legal but professionally questionable will be hard to defend in
a review. An option that is professionally acceptable but personally corrosive will not be
sustained. The table makes the trade-off explicit.

**Branch point.** If an option fails the legal ring, eliminate it. If it fails the professional
ring, escalate. If it fails only the personal ring, document the reasoning and seek support.

### Step 8: Document the decision

**Input:** options and assessment.
**Output:** a dated note with facts, options, rationale, owner, and review date.

**Reasoning.** The note is the artifact that makes the decision reviewable. It should be written
so that a reader who disagrees can see exactly what was known and chosen.

**Branch point.** If the decision cannot be written down, it probably cannot be defended. That is
itself a signal.

### Step 9: Escalate to the level with authority

**Input:** decision note.
**Output:** a decision from someone empowered to make it.

**Reasoning.** The person who notices a problem is rarely the person who can fix it. Escalation
is not an accusation; it is a request for a decision at the right level. The note travels with
the request.

**Branch point.** If the escalation path is blocked, use the next available channel: compliance
hotline, medical safety officer, board, or regulator. Document each attempt.

### Step 10: Act, including the uncomfortable parts

**Input:** decision.
**Output:** implementation with named owners and dates.

**Reasoning.** A decision that is announced but not implemented is worse than no decision,
because it creates a record of awareness without action. The uncomfortable parts, such as
disclosing a limitation to a customer, are usually the parts that matter.

**Branch point.** If implementation is blocked, return to step 9 with the new fact. Blocked
implementation is itself an escalation trigger.

### Step 11: Disclose to those affected

**Input:** decision and implementation.
**Output:** a plain-language disclosure appropriate to the audience.

**Reasoning.** Disclosure is owed to people whose decisions or care may be affected. The form
differs: a clinician needs a factual limitation, a customer needs a contractual notice, a patient
needs an explanation they can understand.

**Branch point.** If disclosure would itself cause harm, consult on timing and form, but do not
substitute silence for consultation.

### Step 12: Review and learn

**Input:** the record.
**Output:** a short retrospective and, where relevant, a change to process.

**Reasoning.** Individual cases are symptoms. The retrospective asks whether the process allowed
the problem to arise, and what would prevent the next one. This is how an organization learns
rather than merely surviving.

**Branch point.** If the same issue recurs, the process change was insufficient. Escalate the
process problem, not only the instance.

### 5.1 The inconvenient result

The most common ethical test in this field is a result that is true and unwelcome. A model works
less well in a subgroup, a pilot shows no benefit, an external validation fails. The professional
obligation is not to like the result. It is to report it accurately and to make the decision with
the result in view.

A useful protocol:

1. Pre-specify the analyses that will be reported, including subgroup analyses.
2. Report the result with its confidence interval, not only its point estimate.
3. State the implication for the claim, not only the number.
4. Distinguish "no evidence of benefit" from "evidence of no benefit".
5. Record who was informed and when.

The distinction in item 4 is the one most often blurred. A pilot with a wide confidence interval
that includes both harm and substantial benefit is not evidence of no benefit. It is evidence
that the study did not answer the question.

### 5.2 Authorship and credit

Authorship disputes are common and corrosive. The International Committee of Medical Journal
Editors (ICMJE) criteria require substantial contribution to conception or design, acquisition or
analysis, drafting or critical revision, final approval, and accountability. All four must be met.

In industry, authorship is often replaced by internal credit and contributor statements. The
professional obligation is the same: credit the people who did the work, and do not accept credit
you did not earn.

| Situation | Professional response |
|---|---|
| Manager added to author list without contribution | Raise it before submission, in writing |
| Junior analyst did the analysis, not listed | Add them or document the contribution |
| Vendor ghost-wrote the methods section | Disclose the support |
| Statistician reviewed the plan but not the paper | Acknowledge, do not list as author |
| Two teams claim the same model | Document contributions by artifact and date |

### 5.3 Conflicts of interest

A conflict exists when a secondary interest could bias a primary professional judgement. The
management options, in order of preference:

1. **Avoid.** Do not take the equity, the advisory role, or the gift.
2. **Divest.** Sell the holding or resign the role.
3. **Recuse.** Remove yourself from the decision.
4. **Disclose.** Make the interest visible to decision-makers and, where appropriate, publicly.
5. **Monitor.** Put oversight in place where disclosure is insufficient.

Disclosure is the weakest option and is often treated as sufficient. For a decision about a
product you own equity in, disclosure does not restore independence.

### 5.4 Data ethics in practice

| Question | Acceptable practice | Unacceptable practice |
|---|---|---|
| Can we use this dataset for a new purpose? | Check consent, DUA, and IRB scope | Assume warehouse access implies permission |
| Can we share this with a vendor? | BAA or DUA, minimum necessary | Send an extract by email |
| Can we publish this? | De-identification, small-cell suppression | Publish row-level data with dates |
| Can we use an LLM on this? | Approved tool, no identifiable data | Paste notes into a public model |
| Can we keep this after the project? | Retention schedule, deletion | Keep indefinitely "just in case" |
| Can we re-identify to improve the model? | Only with approval and a protocol | Attempt it because it is technically possible |

### 5.5 When to escalate outside the organization

Internal escalation is the default. External escalation is appropriate when internal channels
have failed, when there is an imminent risk of serious harm, or when the law requires reporting.
The decision is serious and should be taken with advice. Document every internal attempt first.

External channels include regulators, professional bodies, and, in some jurisdictions, protected
disclosure regimes. The protections differ, and a disclosure made through the wrong channel may
lose legal protection. This is a situation for counsel, not for improvisation.

### 5.6 Designing for honesty

The most effective ethical intervention is structural. If the project is designed so that
inconvenient results are part of the pre-specified output, reporting them becomes routine rather
than heroic.

| Design choice | Effect |
|---|---|
| Pre-register the analysis plan | Prevents outcome switching |
| Pre-specify subgroup analyses | Makes disparities reportable by default |
| Define a negative-result dashboard | Normalizes reporting null findings |
| Separate the evaluation team from the product team | Reduces pressure on the analyst |
| Require a model card before launch | Forces limitation disclosure |
| Give the safety officer a stop authority | Creates a channel independent of the project |
| Record decisions in a dated log | Makes the reasoning reviewable |

None of these guarantees ethical behaviour. They make the ethical choice the default, which is
more reliable than relying on individual courage.

### 5.7 The escalation conversation

Escalation fails more often because of how it is done than because of the concern itself. A
useful structure for the conversation:

1. State the fact, with the calculation or the document.
2. State the risk, in the language of the listener's responsibility.
3. State what you are asking for, specifically.
4. State the timeline, including what happens if there is no decision.
5. Record the outcome and send a summary.

The summary is important. It converts a conversation into a record and gives the decision-maker
a chance to correct the record if you misunderstood.

## 6. Worked example

A heart failure readmission model is ready for deployment. A subgroup analysis shows lower
sensitivity for patients whose preferred language is Spanish. The product lead wants to ship on
schedule. This example carries the decision end to end.

### 6.1 The facts

| Group | Discharges | Sensitivity | Specificity | Prevalence |
|---|---:|---:|---:|---:|
| English-preferring | 1,200 | 0.72 | 0.85 | 20% |
| Spanish-preferring | 300 | 0.58 | 0.85 | 20% |

```text
English cases        = 1,200 x 0.20 = 240
English detected     = 240 x 0.72 = 172.8
English missed       = 240 - 172.8 = 67.2
English non-cases    = 960
English false pos.   = 960 x 0.15 = 144
English alerts       = 172.8 + 144 = 316.8
English PPV          = 172.8 / 316.8 = 0.5455 (54.5%)

Spanish cases        = 300 x 0.20 = 60
Spanish detected     = 60 x 0.58 = 34.8
Spanish missed       = 60 - 34.8 = 25.2
Spanish non-cases    = 240
Spanish false pos.   = 240 x 0.15 = 36
Spanish alerts       = 34.8 + 36 = 70.8
Spanish PPV          = 34.8 / 70.8 = 0.4915 (49.2%)
```

Difference in sensitivity: 0.72 - 0.58 = 0.14, or 14 percentage points.

### 6.2 Is the gap real or noise?

```text
SE of difference = sqrt( p1(1-p1)/n1 + p2(1-p2)/n2 )
                 = sqrt( 0.72 x 0.28 / 1,200 + 0.58 x 0.42 / 300 )
                 = sqrt( 0.000168 + 0.000812 )
                 = sqrt( 0.000980 )
                 = 0.03130

95% CI = 0.14 +/- 1.959964 x 0.03130
       = 0.14 +/- 0.06136
       = 0.0786 to 0.2014
```

The confidence interval excludes zero. The gap is statistically detectable. It is also clinically
material: at the same prevalence, the Spanish group has 25.2 missed cases out of 60, against 67.2
out of 240 in the English group. If the Spanish group had the same sensitivity, it would detect
43.2 cases instead of 34.8, an additional 8.4 detected cases per year.

### 6.3 Is the study powered for smaller gaps?

Suppose the team wants to detect a 5 percentage point gap, 0.72 versus 0.67.

```text
n per group = (1.959964 + 0.841621)^2 x (0.72 x 0.28 + 0.67 x 0.33) / (0.05)^2
            = 7.848878 x (0.2016 + 0.2211) / 0.0025
            = 7.848878 x 0.4227 / 0.0025
            = 3.31772 / 0.0025
            = 1,327.1
            -> 1,328 per group
```

Detecting a 5-point gap requires 1,328 Spanish-preferring patients. The dataset has 300. The
study is underpowered for small gaps, which is why the observed 14-point gap matters: it is large
enough to detect with the available sample.

For a 10-point gap (0.72 versus 0.62), the required sample is 344 per group, which is close to
the available 300. The dataset can detect large gaps but not small ones. Any statement about
equity must therefore be calibrated to that limit.

### 6.4 The options

| Option | Effect on harm | Effect on schedule | Legal and professional status |
|---|---|---|---|
| Ship as planned, no disclosure | Harm continues unmeasured | No delay | Legally possible; professionally indefensible |
| Ship with subgroup monitoring and disclosure | Harm measured; clinicians informed | 2 weeks for monitoring setup | Defensible |
| Delay until more Spanish data are collected | Harm avoided during delay | 6 to 9 months | Defensible but costly |
| Narrow the intended population to English-preferring patients | Harm avoided by exclusion | 2 weeks | Defensible, but creates an access disparity |
| Add a mitigation: translated outreach and clinician review for Spanish-preferring patients | Harm reduced | 4 weeks | Defensible and preferred |

### 6.5 The decision

The team chooses the fifth option with elements of the second: ship with a mitigation, disclose
the limitation in the model card and to clinical leadership, monitor sensitivity by language
quarterly, and set a stop rule if the gap exceeds 10 percentage points for two consecutive
quarters.

The rationale is documented: the mitigation addresses the mechanism (language-concordant
outreach), the disclosure addresses autonomy, the monitoring addresses uncertainty, and the stop
rule addresses the possibility that the mitigation does not work.

### 6.6 The disclosure text

"The model's sensitivity for patients whose preferred language is Spanish was 0.58 in the
validation cohort, compared with 0.72 for English-preferring patients. The difference is
statistically detectable (95% CI for the difference, 7.9 to 20.1 percentage points). A
language-concordant outreach protocol has been added for Spanish-preferring patients. Performance
will be monitored quarterly by language, and the model will be re-evaluated if the gap exceeds 10
percentage points for two consecutive quarters."

This is what a defensible disclosure looks like: specific, quantified, and paired with an action.

### 6.7 What would make the decision wrong

If the mitigation had no plausible mechanism, shipping would be indefensible. If the gap had been
40 points, the model would be unsafe for the subgroup regardless of mitigation. If the hospital
could not monitor, the stop rule would be fictional. The decision depends on those conditions,
which is why they are written down.

## 7. Code, query, or calculation

The block computes subgroup performance, the confidence interval for the difference, and the
sample size needed to detect smaller gaps. Standard library only.

```python
"""Subgroup gap analysis and power for a two-proportion comparison."""

from dataclasses import dataclass
from math import ceil, sqrt


@dataclass(frozen=True)
class Group:
    name: str
    n: int
    prevalence: float
    sensitivity: float
    specificity: float


def metrics(g: Group) -> dict:
    cases = g.n * g.prevalence
    non_cases = g.n - cases
    tp = cases * g.sensitivity
    fn = cases - tp
    fp = non_cases * (1 - g.specificity)
    tn = non_cases - fp
    alerts = tp + fp
    return {
        "cases": cases, "tp": tp, "fn": fn,
        "fp": fp, "tn": tn, "alerts": alerts,
        "ppv": tp / alerts if alerts else float("nan"),
    }


def diff_ci(p1: float, n1: int, p2: float, n2: int, z: float = 1.959964) -> dict:
    diff = p1 - p2
    se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {
        "diff": diff, "se": se,
        "low": diff - z * se,
        "high": diff + z * se,
        "z_stat": diff / se,
    }


def n_per_group(p1: float, p2: float, z_alpha: float = 1.959964,
                z_beta: float = 0.841621) -> int:
    num = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    return int(ceil(num / (p1 - p2) ** 2))


eng = Group("English", 1_200, 0.20, 0.72, 0.85)
spa = Group("Spanish", 300, 0.20, 0.58, 0.85)

for g in (eng, spa):
    m = metrics(g)
    print(f"{g.name}: cases {m['cases']:.1f}, detected {m['tp']:.1f}, "
          f"missed {m['fn']:.1f}, alerts {m['alerts']:.1f}, PPV {m['ppv']:.4f}")

ci = diff_ci(0.72, 1_200, 0.58, 300)
print(f"\nDifference {ci['diff']:.4f}")
print(f"SE {ci['se']:.5f}")
print(f"95% CI {ci['low']:.4f} to {ci['high']:.4f}")
print(f"z {ci['z_stat']:.2f}")

print(f"\nN per group for 5-point gap: {n_per_group(0.72, 0.67):,}")
print(f"N per group for 10-point gap: {n_per_group(0.72, 0.62):,}")

detected_if_equal = 60 * 0.72
print(f"Additional cases detected if sensitivity were equal: "
      f"{detected_if_equal - 34.8:.1f}")
```

Expected output:

```text
English: cases 240.0, detected 172.8, missed 67.2, alerts 316.8, PPV 0.5455
Spanish: cases 60.0, detected 34.8, missed 25.2, alerts 70.8, PPV 0.4915

Difference 0.1400
SE 0.03130
95% CI 0.0786 to 0.2014
z 4.47

N per group for 5-point gap: 1,328
N per group for 10-point gap: 344
Additional cases detected if sensitivity were equal: 8.4
```

Three checks. First, the 95% interval is 0.14 plus or minus 0.0614, which gives 0.0786 to 0.2014.
Second, the 5-point gap needs 1,328 patients per group, more than the 300 available. Third, the
10-point gap needs 344 per group, which is close to the available sample, so the dataset can
detect large gaps but not small ones.

> **Note:** The normal approximation is adequate for these proportions and sample sizes. For small
> subgroups, use an exact method or a statistician's calculation.

## 8. Data, standards, and regulatory context

| Instrument | Issuer | What it constrains | Status |
|---|---|---|---|
| Common Rule, 45 CFR 46 | US federal agencies | Human-subjects research, consent, IRB review | Binding for covered research |
| HIPAA Privacy Rule | US HHS | Use and disclosure of PHI | Binding |
| GDPR, Regulation (EU) 2016/679 | EU | Personal data processing, consent, automated decisions | Binding in the EU |
| GDPR Article 22 | EU | Solely automated decisions with significant effects | Binding, with exceptions |
| EU AI Act, Regulation (EU) 2024/1689 as amended | EU | Transparency, human oversight, risk management | Binding on phased timeline |
| Declaration of Helsinki | World Medical Association | Ethical principles for human research | Professional standard |
| Belmont Report | US National Commission | Principles for human-subjects research | Foundational, not binding itself |
| ICMJE recommendations | ICMJE | Authorship, disclosure, reporting | Journal requirement |
| COPE guidance | COPE | Publication ethics, retractions, corrections | Publisher membership standard |
| ORI misconduct definition | US ORI | Fabrication, falsification, plagiarism | Binding for federally funded research |
| Sunshine Act | US CMS | Reporting of payments to physicians | Binding |
| FDA regulations on financial disclosure | US FDA | Disclosure of clinical investigator interests | Binding for submissions |

### 8.1 What the rules do not cover

Most of the daily ethical questions in data science are not covered by a specific rule. There is
no regulation that says a model with a 14-point subgroup gap must not ship. The obligation comes
from professional judgement, institutional policy, and the general duty to avoid foreseeable harm.

That is why the procedure in section 5 matters. In the absence of a rule, the process is the
protection: documented facts, consultation, options, decision, disclosure.

### 8.2 GDPR Article 22

Article 22 restricts decisions based solely on automated processing that produce legal effects or
similarly significantly affect a person. It applies to some clinical and insurance decisions and
not to others. Where it applies, the data subject has rights to human intervention, to express a
view, and to contest the decision.

A model that flags patients for outreach is usually not a solely automated decision with legal
effect. A model that denies coverage might be. The distinction is fact-specific and requires
counsel.

### 8.3 The AI Act and professional conduct

The AI Act adds obligations that overlap with professional ethics: human oversight, transparency,
data governance, and post-market monitoring. For a high-risk system, the provider must implement
a risk management system and ensure that natural persons overseeing the system can understand its
outputs. The amended timeline applies high-risk obligations from 2 December 2027 for Annex III
systems and 2 August 2028 for Annex I products.

This is orientation, not legal advice.

### 8.4 Professional codes

The ACM Code of Ethics, the AMA principles of medical ethics, and the codes of professional
societies in informatics and biostatistics all contain overlapping obligations: avoid harm,
be honest, protect privacy, disclose conflicts, and maintain competence. They are not enforceable
in the way law is, but they define what a peer can hold you to.

## 9. Industry practice and case evidence

### 9.1 The cost-proxy algorithm

**Documented.** Obermeyer and colleagues, in Science in 2019, showed that a widely used population
health algorithm used health care costs as a proxy for need. At a given risk score, Black patients
were substantially sicker than White patients. Correcting the label would have increased the
share of Black patients identified for additional care from 17.7% to 46.5%.

**Inference.** The case is an ethics failure with a technical cause. The team measured access to
care rather than need for care. No individual acted in bad faith; the label was convenient and
the outcome was a disparity at scale.

### 9.2 The sepsis model

**Documented.** Wong and colleagues published an external validation of a widely implemented
proprietary sepsis prediction model in JAMA Internal Medicine in 2021. The model's AUC was
approximately 0.63 in the validation cohort, and at the operating threshold it identified about
one third of patients who developed sepsis.

**Inference.** The ethical issue is not that the model was imperfect. It is that a product was
deployed at scale before independent validation in the populations where it was used. The
professional obligation to validate before deployment is the lesson.

### 9.3 Research integrity at scale

**Documented.** A 2025 study in PNAS by Richardson and colleagues, using Retraction Watch,
PubPeer, Web of Science, Scopus, OpenAlex, and the Problematic Paper Screener, reported that
suspected paper-mill products are doubling roughly every 1.5 years, while total publications
double roughly every 15 years. Only 28.7% of suspected paper-mill products had been retracted.
The authors estimated that only about 25% would ever be retracted and about 10% would ever sit in
a deindexed journal.

**Inference.** The integrity problem is systemic, not individual. For a data scientist, the
practical implication is that literature reviews must include an integrity check, and that
citation counts are not a measure of reliability.

### 9.4 Governance maturity in 2025

**Documented.** A 2025 survey by the UPMC Center for Connected Medicine with KLAS Research,
covering 31 respondents, reported that only 16% had a system-wide AI governance policy. A 2025
HFMA and Eliciting Insights survey of 233 health system executives reported that 18% had a mature
governance structure and a fully formed AI strategy, while 29% had no AI governance structure and
88% used AI in some form.

**Inference.** Most organizations do not yet have a committee that will catch an ethical problem
before it ships. The individual professional is therefore the control of last resort, which is
exactly the situation this topic prepares you for.

### 9.5 What professional conduct looks like in practice

**Inference, from public accounts of well-run programs.** Organizations that handle these
situations well tend to share four practices: a pre-specified analysis plan that includes
subgroups; a standing safety review with authority to stop; a disclosure template that requires
quantified limitations; and a decision log that is reviewed at project closure. None is
glamorous. All are auditable.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Shipping a known subgroup gap without disclosure | Subgroup table exists but is not in the deck | Commercial pressure | Harm and liability | Disclose, mitigate, monitor |
| Outcome switching | Primary endpoint changes after results | Convenience | Invalid claim | Pre-registration |
| p-hacking | Many analyses, one reported | Selective reporting | False positive | Pre-specified plan |
| HARKing | Hypothesis appears after results | Narrative bias | Misleading literature | Label exploratory analyses |
| Gift authorship | Senior name added without contribution | Hierarchy | Credit injustice | ICMJE criteria in writing |
| Ghost authorship | Vendor writes methods, not disclosed | Convenience | Integrity breach | Disclose support |
| Undisclosed COI | Equity or advisory role omitted | Conflict | Biased judgement | Annual disclosure |
| Secondary use without basis | New analysis on old data | Assumption of permission | Privacy breach | Check DUA and consent |
| Re-identification attempt | "Just to check" | Technical curiosity | Legal and ethical breach | Prohibit; audit access |
| Pasting PHI into a public LLM | Convenience | Data leakage | Breach | Approved tools only |
| Suppressing a negative result | "Not ready to share" | Career pressure | Publication bias | Negative-result reporting |
| Escalation without record | Verbal complaint only | Fear of conflict | No institutional memory | Written summary |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Facts can be separated from inference | Credible escalation | Ambiguous evidence | Reproduce the calculation |
| Colleagues will engage with a documented concern | Institutional process | Culture of avoidance | Ask for a written response |
| Rules cover the important cases | Compliance | Gaps in regulation | Consult ethics or counsel |
| Disclosure reduces harm | Autonomy | Disclosure without action | Pair with mitigation |
| Subgroup analyses are powered | Equity claims | Small samples | Report confidence intervals |
| Conflicts can be managed | Governance | Structural conflict | Recusal or divestment |
| Escalation channels exist | Accountability | No channel | Identify before needed |
| The record will be read | Deterrence | No audit | Write for a future reader |

### 11.1 Limits

This topic describes professional expectations in the US and EU. Other jurisdictions have
different rules and different professional norms.

The worked example uses constructed numbers. The arithmetic is exact; the values are not
transferable.

The procedure in section 5 is a heuristic. It is not a substitute for an ethics committee, and it
does not guarantee that the right decision will be available.

## 12. Exercises

### Exercise 1: Name the pressure (recall, 20 minutes)

**Deliverable.** Four short scenarios, each illustrating one of the four pressures, with the
moment at which judgement could be distorted.

**Acceptance criteria.** Each scenario names a specific decision, not a general attitude. The
distortion point is identifiable.

**Time.** 20 minutes.

### Exercise 2: Misconduct or QRP (application, 25 minutes)

**Deliverable.** Ten practices classified as misconduct, QRP, or acceptable, with a one-line
rationale.

**Acceptance criteria.** Uses the federal misconduct definition. Distinguishes honest error from
QRP. No practice classified without a reason.

**Time.** 25 minutes.

### Exercise 3: Escalation plan (application, 30 minutes)

**Deliverable.** A twelve-step escalation plan for a concern about a model with a known safety
issue.

**Acceptance criteria.** Each step has an owner and an artifact. At least two escalation channels
identified. A timeline is included.

**Time.** 30 minutes.

### Exercise 4: Subgroup analysis (calculation, 40 minutes)

**Deliverable.** For a subgroup with 250 patients, sensitivity 0.61 versus 0.74 in 1,100
patients, compute the difference, standard error, 95% CI, and the sample needed to detect a
5-point gap.

**Acceptance criteria.** Arithmetic shown. Interpretation distinguishes statistical from clinical
significance. States whether the available sample is adequate.

**Time.** 40 minutes.

### Exercise 5: Authorship dispute (critique, 25 minutes)

**Deliverable.** A written response to a manager who was added as an author without meeting ICMJE
criteria.

**Acceptance criteria.** Cites the criteria. Proposes a resolution that preserves the relationship.
Does not accuse.

**Time.** 25 minutes.

### Exercise 6: Disclosure statement (design, 30 minutes)

**Deliverable.** A 150-word disclosure for a model with a known limitation, suitable for a model
card.

**Acceptance criteria.** Quantified limitation. Population and setting. Mitigation. Monitoring
plan. No marketing language.

**Time.** 30 minutes.

### Exercise 7: COI management (application, 25 minutes)

**Deliverable.** A COI management plan for an analyst who holds equity in a vendor being
evaluated.

**Acceptance criteria.** Chooses among avoid, divest, recuse, disclose, monitor. Justifies the
choice. Identifies who else must know.

**Time.** 25 minutes.

### Exercise 8: Design for honesty (design, 30 minutes)

**Deliverable.** Three structural changes to a project that would make an inconvenient result
reportable by default.

**Acceptance criteria.** Each change is specific and implementable. Each identifies the failure it
prevents.

**Time.** 30 minutes.

## 13. Self-check answers

**1. What are the four pressures?**
Commercial, career, institutional, and time.

**2. Why is naming the specific fact important?**
Because a specific fact can be examined, reproduced, and responded to. A general feeling cannot.

**3. What is the difference between misconduct and QRP?**
Misconduct is fabrication, falsification, or plagiarism. QRP distorts the literature without
meeting those definitions, for example p-hacking or outcome switching.

**4. What are the three rings?**
Legal, professional, and personal.

**5. Compute the sensitivity difference and its standard error for 0.72 in 1,200 and 0.58 in 300.**
Difference 0.14; SE 0.0313; 95% CI 0.0786 to 0.2014.

**6. Why does the 5-point gap require more patients than the 14-point gap?**
Because the required sample is inversely proportional to the square of the difference. Smaller
differences need much larger samples.

**7. What is the weakest COI management option?**
Disclosure. It makes the interest visible but does not restore independence.

**8. When is external escalation appropriate?**
When internal channels have failed, when there is imminent risk of serious harm, or when the law
requires reporting.

**9. Why is "no evidence of benefit" different from "evidence of no benefit"?**
The first describes an underpowered or inconclusive study; the second requires a study precise
enough to exclude a meaningful effect.

**10. What makes a disclosure defensible?**
It is specific, quantified, states the population and setting, and is paired with an action and a
monitoring plan.

### 13.1 Rubric for exercises 4 and 6

Exercise 4 is scored on arithmetic (0 to 3), interpretation (0 to 3), and the adequacy judgement
(0 to 2). An answer that reports the difference without a confidence interval scores at most 4.

Exercise 6 is scored on quantification, clarity, presence of a mitigation, and absence of
marketing language.

## 14. Interview questions

**1. "Tell me about a time you found a result you did not want."**
Strong: describes the fact, the pressure, the process, and the disclosure. Weak: describes only
the technical fix.

**2. "What would you do if a leader asked you to remove a subgroup analysis?"**
Strong: asks why, documents the request, escalates, and refuses to remove a known limitation.
Weak: complies or confronts without a record.

**3. "What is research misconduct?"**
Strong: fabrication, falsification, plagiarism, and the distinction from honest error. Weak:
"making mistakes".

**4. "How do you handle authorship?"**
Strong: ICMJE criteria, written agreement early, contributorship statement. Weak: "by seniority".

**5. "How would you escalate a safety concern?"**
Strong: fact, risk, ask, timeline, record, channel. Weak: "tell my manager".

**6. "What is a conflict of interest?"**
Strong: a situation, not a behaviour; disclose and manage; non-financial conflicts count. Weak:
"money".

**7. "When can you use patient data for a new project?"**
Strong: check consent, DUA, IRB scope, and minimum necessary. Weak: "if it is de-identified".

**8. "How do you decide whether a subgroup gap is real?"**
Strong: confidence interval, sample size, clinical materiality, and a pre-specified analysis.
Weak: compares two percentages.

**9. "What would you do if the model cannot be fixed before launch?"**
Strong: disclose, mitigate, monitor, stop rule, or delay. Weak: "ship and fix later".

**10. "How do you use LLMs without breaching privacy?"**
Strong: approved tools, no identifiable data, contractual safeguards, disclosure of use. Weak:
"it is fine, it is private".

**11. "What does professional conduct mean in a commercial setting?"**
Strong: honesty about limitations, documented decisions, disclosure of conflicts, protection of
affected people. Weak: "following the law".

**12. "What would make you leave a project?"**
Strong: a specific harm that cannot be mitigated, a request to falsify, or a blocked escalation.
Weak: "a better offer".

## 15. Key terms

- **Allocative harm** — unequal distribution of resources or attention.
- **Authorship** — attribution for substantive contribution.
- **Belmont Report** — US principles for human-subjects research.
- **COI** — conflict of interest.
- **COPE** — Committee on Publication Ethics.
- **Correction** — fixing an error in published work.
- **De-identification** — removal of identifiers under a standard.
- **DSMB** — data and safety monitoring board.
- **Duty to document** — obligation to record decisions.
- **Duty to warn** — obligation to inform a person at risk.
- **Escalation** — raising a concern to a level with authority.
- **Fabrication** — inventing data or results.
- **Falsification** — manipulating or omitting data.
- **Ghost authorship** — omitted contributing author.
- **Gift authorship** — authorship without qualifying contribution.
- **HARKing** — hypothesizing after results are known.
- **ICMJE** — authorship and disclosure standards body.
- **Informed consent** — voluntary agreement based on adequate information.
- **IRB** — institutional review board.
- **Minimum necessary** — limit data to what is needed.
- **Misconduct** — fabrication, falsification, plagiarism.
- **Model card** — structured model documentation.
- **ORI** — US Office of Research Integrity.
- **Outcome switching** — changing the primary outcome after the study.
- **p-hacking** — selective analysis for significance.
- **Plagiarism** — using others' work without attribution.
- **Privacy harm** — loss of control over personal information.
- **Professional conduct** — expected behaviour of a profession.
- **QRP** — questionable research practice.
- **Re-identification** — recovery of identity from de-identified data.
- **Representational harm** — stereotyping or erasure.
- **Research misconduct** — see misconduct.
- **Retraction** — withdrawal of a published work.
- **Secondary use** — use for a purpose other than the original.
- **Sunshine Act** — US payment transparency law.
- **Undue influence** — pressure compromising independent judgement.
- **Whistleblowing** — reporting wrongdoing outside the chain of command.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- World Health Organization, *Ethics and governance of artificial intelligence for health:
  Guidance on large multi-modal models*, 2024, ISBN 978-92-4-008475-9.
  https://www.who.int/publications/i/item/9789240084759
- Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S., "Dissecting racial bias in an
  algorithm used to manage the health of populations," *Science*, 2019.
  https://doi.org/10.1126/science.aax2342
- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction
  Model in Hospitalized Patients," *JAMA Internal Medicine*, 2021.
- Richardson, K., Hong, K., Byrne, J., Stoeger, T., Amaral, L., "The entities enabling scientific
  fraud at scale are large, resilient, and growing rapidly," *PNAS*, 2025.
  https://doi.org/10.1073/pnas.2420092122
- International Committee of Medical Journal Editors, Recommendations for the Conduct, Reporting,
  Editing, and Publication of Scholarly Work in Medical Journals.
  https://www.icmje.org/recommendations/
- Committee on Publication Ethics, Core Practices and guidance on retractions and corrections.
  https://publicationethics.org/
- US Office of Research Integrity, Definition of Research Misconduct, 42 CFR Part 93.
  https://ori.hhs.gov/
- US Department of Health and Human Services, 45 CFR Part 46, Protection of Human Subjects.
- US Department of Health and Human Services, HIPAA Privacy Rule.
  https://www.hhs.gov/hipaa/
- European Parliament and Council, Regulation (EU) 2016/679 (GDPR), Articles 6, 9, 22, 35.
- European Parliament and Council, Regulation (EU) 2024/1689 (AI Act), as amended in 2026.
- World Medical Association, Declaration of Helsinki, 1964 with subsequent amendments.
  https://www.wma.net/
- The National Commission for the Protection of Human Subjects of Biomedical and Behavioral
  Research, The Belmont Report, 1979.
- UPMC Center for Connected Medicine and KLAS Research, "How Health Systems Are Navigating the
  Complexities of AI," 2025.
- Healthcare Financial Management Association and Eliciting Insights, "Health System Readiness for
  Artificial Intelligence," 2025.
- US Centers for Medicare and Medicaid Services, Physician Payments Sunshine Act.
  https://www.cms.gov/priorities/key-initiatives/open-payments

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_regulatory-triage-map.md`](07_regulatory-triage-map.md)
- Next topic: [`09_how-to-use-this-package.md`](09_how-to-use-this-package.md)
- Related topic: [`03_stakeholder-and-buyer-map.md`](03_stakeholder-and-buyer-map.md)
- Related topic: [`10_success-metrics-and-portfolio.md`](10_success-metrics-and-portfolio.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../course/00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
