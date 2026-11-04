# Multimodal: Shelf Images, Damage, and Video

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic arithmetic and spreadsheet use; evaluation concepts are introduced below  
> Version 2026.5

## 1. Why this topic exists

A retail associate cannot check every shelf continuously.
A store may have stock in the back room while an assigned shelf position is empty.
A photograph can suggest that a facing needs attention,
but it cannot establish the cause of the gap or prove a sale was lost.
The useful decision is whether to send an associate to check and replenish,
not whether a model can describe a photograph fluently.

At a distribution center, a damaged carton may indicate a risk to its contents.
A camera at receiving can route a suspect carton for inspection,
while an operator looks inside or follows the applicable quality procedure.
A dent in packaging does not automatically establish product damage,
carrier liability, food safety, or eligibility for a claim.

On a production line, a missing seal can be more serious than a scuffed label.
A visual model may help find visible defects,
but an image does not replace measurement of hidden defects,
required destructive tests, or an approved release procedure.
The defect's consequences determine the required evidence and response time.

Video can show a sequence that one image misses,
for example a package moving through a camera's field of view.
Consecutive frames are highly similar, however,
so thousands of frames are not thousands of independent observations.
Cameras can also capture identifiable customers and workers.
A useful inspection design limits collection and prevents unintended surveillance.

A multimodal system combines two or more forms of information,
such as an image, a timestamp, a planogram, and inventory records.
The joining keys and capture times matter as much as model sophistication.
An old planogram can turn a correct image reading into a wrong action.
A language model that explains a picture may sound certain even when it is wrong.

This unit starts with the business decision and works backward to evidence,
labels, capture design, evaluation, unit economics, privacy, and rollout.
The case and all numerical results are fictional teaching assumptions,
not reported results for any company or deployed system.
Module 06 is not required: labels, data splits, precision, recall,
selection bias, and uncertainty are explained where they are used.

## 2. Learning outcomes

After this unit you can:

1. Define an observable inspection decision and distinguish image evidence
   from inventory state, physical product condition, and authorized action.
   Exercise 1 tests the boundary.
2. Select between classification, detection, segmentation, optical character
   recognition, anomaly detection, and human inspection for a stated task.
   Exercise 2 requires a model and data contract.
3. Design capture and annotation for shelves, cartons, and short video clips,
   including occlusion, unknowns, labeling disagreements, and privacy.
   Exercises 2 and 3 examine these choices.
4. Compute a confusion matrix, precision, recall, and workload at a given
   prevalence and explain why a test sample may not represent operations.
   Exercise 4 requires the calculations.
5. Estimate the incremental economics of an assisted inspection workflow,
   including missed defects, false alerts, labor, and fixed costs.
   Exercise 5 tests a launch decision.
6. Design a leakage-resistant evaluation, a prospective pilot, and a monitoring
   and rollback plan with event-level rather than frame-level outcomes.
   Exercise 6 evaluates the design.
7. Identify privacy, security, quality, and vendor-management controls
   appropriate to a particular deployment context.
   Exercises 7 and 8 apply the controls.

Successful use may mean choosing a simpler camera rule or retaining manual
inspection when the model does not improve the decision at acceptable risk.

## 3. The core mental model

```text
Decision and consequence of error
             |
             v
Capture design + permission + approved field of view
             |
             v
Image / clip + time + location + image-quality status
             |
             +----> unreadable or out-of-scope -> recapture / human route
             v
Versioned context: SKU, planogram, product and defect rules
             |
             v
Model proposes region, class, score and evidence reference
             |
             v
Validate identity, freshness, quality and business rules
             |
             v
Human checks evidence or an approved low-risk action is triggered
             |
             v
Replenish / inspect / hold / no action, with an audit trail
             |
             v
Independently sampled outcomes -> evaluation -> monitoring -> rollback
```

The image is evidence about visible surfaces at a particular moment.
The detection is a hypothesis about that evidence.
A reviewer decision is not the same thing as the original image.
A stock transfer, product release, or claim is a separate authorized action.
Keep those records separate so later corrections do not erase the evidence.

A shelf has physical positions, not just pixels.
A planogram describes intended item placement and facing allocation.
The image shows only what the camera can see.
A stock ledger describes recorded on-hand quantity, which can be stale.
Together they may support an investigation;
none alone proves why a product is missing or whether it is available elsewhere.

A carton has outer packaging, inner packaging, and product contents.
A vision system might locate a tear in the visible outer surface.
The quality owner decides whether that defect requires quarantine,
inspection of contents, repacking, or another prescribed disposition.
A hidden puncture cannot be detected if it is outside the camera's view.

A video clip adds temporal order and multiple views,
but the decision unit should normally be a unique package,
a shelf visit, or a defined event rather than each frame.
Track an object across frames only within validated operating conditions.
A track identity can split or merge under occlusion,
so deduplication and a fallback to human review are necessary.

A forecast is an estimate of a future quantity;
a plan is an intended course of action;
a target is a desired outcome;
a commitment is an authorized promise or obligation.
For example, projected future alerts are a forecast,
a pilot staffing roster is a plan,
a 15-minute median response is a target,
and a signed service obligation is a commitment.
An image-derived alert is none of these by itself.

The loop is valuable only if it changes a decision in time.
A highly accurate alert arriving after a shelf reset or package dispatch
may have little operational value.
Likewise, a fast alert that overwhelms staff may reduce attention to true defects.
Measure the complete workflow, not just the model.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Inspection unit | Physical item, shelf position, or event receiving one decision | Treating each video frame as a new item |
| Image | A recorded two-dimensional array of pixel values at a time | Assuming it exposes hidden surfaces |
| Video clip | Time-ordered frames over a bounded interval, measured in seconds | Treating adjacent frames as independent cases |
| Frame rate | Number of frames captured or processed per second | Assuming more frames always improve decisions |
| Field of view | Physical area visible to a camera | Assuming an obscured region was checked |
| Resolution | Pixel dimensions of an image or image region | Equating megapixels with sufficient defect detail |
| Ground sampling distance | Approximate physical distance per pixel at an object plane, such as mm/pixel | Applying one distance to a scene with large depth changes |
| Blur | Loss of image detail from motion, focus, or other causes | Calling blurred images negative examples |
| Occlusion | Blocking of a target by another object or person | Interpreting the blocked target as absent |
| Planogram | Versioned representation of intended shelf placement | Treating a plan as proof of present shelf state |
| Facing | Visible front presentation of a product on a shelf | Counting facings as units of back-room inventory |
| Stock keeping unit (SKU) | Operational identifier for a distinct stocked item | Assuming a package image uniquely identifies every SKU |
| Shelf availability | Defined eligible position visibly contains its assigned item at a specified instant | Treating availability as total inventory or fulfilled demand |
| Out-of-stock event | Operationally defined period or check in which an item is unavailable | Calling every visual gap a confirmed stockout |
| Label | Annotated target value for a unit or region | Assuming any one annotator is always correct |
| Annotation policy | Written rules defining classes, evidence, boundaries, and unknowns | Leaving ambiguous cases to individual intuition |
| Classification | Assigning a category to an image or crop | Assuming it localizes the relevant object |
| Object detection | Locating and classifying objects, often with bounding boxes | Assuming a box explains the model's reasoning |
| Segmentation | Assigning a class to pixels or regions | Assuming a precise outline proves the correct class |
| Bounding box | Rectangle around a labeled object, in image coordinates | Treating an approximate box as physical dimensions |
| Intersection over union (IoU) | Area of box intersection divided by area of union | Using a matching threshold without reporting it |
| Tracking | Linking detections thought to represent one object across frames | Assuming a track always equals one physical object |
| Optical character recognition (OCR) | Extraction of printed or handwritten text from an image | Treating recognized text as a verified identifier |
| Barcode | Machine-readable encoded identifier under an applicable symbology | Assuming a scanned code alone proves physical condition |
| Anomaly detection | Scoring observations that differ from a learned or defined normal pattern | Equating unusual appearance with a harmful defect |
| Foundation vision model | Model pretrained broadly and adapted or prompted for a visual task | Treating broad pretraining as site-specific validation |
| Vision-language model | Model processing images and text to generate or rank outputs | Treating fluent text as a verified observation |
| Multimodal join | Link between image evidence and records such as planogram or shipment | Joining by an ambiguous or stale identifier |
| Score | Numeric model output used for ranking or thresholding | Treating every score as a calibrated probability |
| Threshold | Score cutoff used to select an alert, usually dimensionless | Assuming a default threshold is optimal everywhere |
| Calibration | Agreement between predicted probabilities and observed rates in a population | Confusing calibration with discrimination |
| Abstention | Explicit decision not to infer a class when evidence or confidence is inadequate | Silently treating unknown as negative |
| Reference standard | Best available independent determination of the actual condition | Using the model's own prediction as truth |
| Prevalence | Share of eligible inspection units that truly have the target condition | Assuming a deliberately balanced test sample reflects operations |
| True positive (TP) | Alert on a unit confirmed to have the target condition | Counting repeated alerts as separate packages |
| False positive (FP) | Alert on a unit without the target condition under the reference policy | Calling an ambiguous reference case definitively wrong |
| False negative (FN) | No alert on a unit that truly has the condition | Measuring only reviewed alerts and overlooking misses |
| True negative (TN) | No alert on a unit without the condition | Assuming all unreviewed cases are negative |
| Precision | TP divided by TP plus FP, among alerts | Assuming precision transfers unchanged across prevalence |
| Recall or sensitivity | TP divided by TP plus FN, among actual positives | Reporting recall without sampling non-alerts |
| Specificity | TN divided by TN plus FP, among actual negatives | Ignoring the false alerts implied by high negative volume |
| False-positive rate (FPR) | FP divided by FP plus TN | Mistaking a small rate for a small absolute workload |
| Data leakage | Information in model development or evaluation that would not be available at the decision point | Splitting near-duplicate frames randomly across sets |
| Drift | Change in inputs, relationships, or outcomes over time | Assuming every seasonal change is model degradation |
| Human-in-the-loop | A person performs a specified decision or verification step | Calling a rubber-stamp queue meaningful oversight |
| Latency | Elapsed time between capture and actionable result, in seconds or minutes | Measuring only model inference and ignoring queues |
| Incremental value | Difference in economic outcome relative to a specified alternative | Counting gross detected value as net benefit |

The relevant denominator always follows the decision.
Precision for a frame alert can be very different from precision for a unique carton.
State the unit, time window, reference policy, and matching rule with every score.

## 5. The mechanics, step by step

### 5.1 Name the operational decision

**Input:** a business problem, existing workflow, and decision owner.
**Output:** a written decision contract and a feasible alternative.

1. Specify who acts, on what physical unit, and by what deadline.
2. Name the consequence of a miss, unnecessary alert, and unreadable image.
3. Identify the existing manual procedure and its measured cost and performance.
4. Identify any mandatory quality, safety, labor, or privacy restrictions.
5. State whether the model assists a person or controls a low-risk action.
6. Define a safe path when an image or contextual record is missing.

For shelves, one decision can be: inspect one assigned facing within 15 minutes
when the camera suggests the assigned item is absent.
For receiving, one decision can be: send a visible suspect carton to a quality
station before put-away, rather than automatically rejecting a shipment.
For a seal, the decision may be to hold a unit pending a prescribed inspection.

These are different products even if they share a camera.
Their labels, tolerances, action authority, and economics differ.
Do not use a shelf benchmark to establish safety of a production release.

A simple alternative might be a barcode check, a fixed sensor,
a visual sampling route, a rules engine, or a second manual inspector.
Compare against the actual alternative, not against doing nothing.
A manual process can also miss defects and has its own burden.

### 5.2 Define the population and unit

**Input:** decision contract and operating calendar.
**Output:** eligible population, sampling frame, and exclusion policy.

1. List sites, shifts, item types, camera positions, and operating hours.
2. Define the decision unit: one eligible shelf-position check,
   one carton presented at receiving, or one product passing a camera.
3. Specify whether repeated checks of the same unit count separately.
4. Establish how a package gets a stable identifier without guessing identity.
5. Define missing images, blocked views, and duplicates as explicit states.
6. Record all eligible units, not only images on which a model returns a score.

An eligible shelf check might require a current assignment and an open store.
A seasonal display not represented in the planogram can be out of scope.
The proportion excluded is itself a performance measure.
Otherwise a system can appear excellent by declining difficult cases silently.

For video, create one event identifier per passage through a defined zone.
If a carton is visible in 50 frames, it still represents one carton.
If it stops and returns, a documented tracking and business rule determines
whether this is one event or two.
A manual override must be available when the camera cannot distinguish them.

### 5.3 Design capture around the smallest relevant defect

**Input:** physical target, smallest defect of concern, throughput, and environment.
**Output:** camera and lighting specification with acceptance tests.

1. Identify the surface and angle needed to see the target.
2. Estimate the required physical detail in millimeters (mm).
3. Relate field width in mm to image width in pixels at the target plane.
4. Choose adequate illumination, shutter speed, focus, and mechanical stability.
5. Verify performance at the fastest operating speed and deepest location.
6. Test glare, shadows, wrapping film, dust, vibration, and changing ambient light.
7. Decide whether a second view or a physical turn is needed.
8. Specify how an operator handles a failed image-quality check.

For an illustrative 600 mm field width and 1,200 usable horizontal pixels,
the approximate scale is 600 / 1,200 = 0.5 mm per pixel at that plane.
A 2 mm mark spans about 4 pixels horizontally under ideal alignment.
That is not a guarantee of detection: contrast, optics, motion, and compression
can erase the mark, and objects at other depths have different scale.
Validate with physical samples of known size and relevant packaging variation.

If a belt moves 800 mm per second and exposure is 1/400 second,
the uncorrected object displacement during exposure is about 2 mm.
At 0.5 mm per pixel that is approximately 4 pixels of motion.
Shorter exposure may require more light; rolling shutter and processing
may add distortion, so measure actual images rather than relying on the estimate.

For shelves, price labels, reflections, customers, restocking carts,
and adjacent product facings can block or confuse a camera.
For cartons, a top camera does not see the underside.
Do not infer a complete inspection from a partial view.
Mark the missing surface as not observed and route according to the quality plan.

### 5.4 Define annotation and reference evidence

**Input:** capture samples, defect policy, and quality-owner expertise.
**Output:** versioned annotation manual and adjudicated reference labels.

1. Write observable criteria for each class and severity.
2. Define object boundaries and permissible box or mask overlap.
3. Include examples of confusing but acceptable cases.
4. Provide explicit unknown, not visible, and out-of-scope labels.
5. Define which evidence can settle each dispute, including physical inspection.
6. Train annotators on a shared calibration set and record disagreements.
7. Adjudicate disputes without allowing the model output to become the answer.
8. Record label version, reviewer, and supporting evidence separately.

A shelf-gap label might mean that the assigned facing is visible and empty.
It must not include a region hidden behind a shopper or camera glare.
An empty location assigned to another SKU is a planogram or assignment issue,
not automatically an out-of-stock event for the visually adjacent item.

A damage label might distinguish cosmetic scuff, torn outer carton,
visible inner packaging exposure, and product damage confirmed on inspection.
The last class cannot be inferred from the outer image alone unless its
reference policy explicitly permits that inference for a defined condition.
A reviewer can conclude that an image is insufficient.

On a production line, separate a visibly missing seal from a failed leak test.
A closed-looking seal may still leak.
If the business outcome is seal integrity, collect the appropriate physical
reference measurement and acknowledge incomplete visibility.
The reference standard may itself be imperfect; preserve uncertainty.

### 5.5 Link context without inventing identity

**Input:** timestamped capture, location or device ID, and approved source records.
**Output:** evidence record with resolved, ambiguous, or missing context.

1. Synchronize clocks and record the time zone and clock-health status.
2. Resolve camera location to a site, aisle, shelf, belt, or inspection station.
3. Join the planogram version that was valid at the image time.
4. Link carton or product identity using an approved scanner or verified key.
5. Keep join provenance, source timestamp, and version identifiers.
6. Mark unresolved or conflicting joins; do not substitute a plausible match.
7. Apply access controls to both images and joined operational data.

A barcode can identify a carton or SKU if it is readable and the mapping is valid.
OCR can read a printed lot code, but a check digit or system lookup may be needed.
Neither a language model's confident guess nor nearby shelf text is a reliable key.
If the source records disagree, route to review or recapture.

A record may include an image ID, event ID, captured-at time, device ID,
location ID, planogram version, SKU key, quality status, model version,
score, proposed label, reviewer outcome, and action timestamp.
Record a link to approved evidence rather than duplicating unrestricted footage.

### 5.6 Choose a model for the question

**Input:** label policy, image quality, operational deadline, and baseline results.
**Output:** candidate methods and a falsifiable evaluation plan.

A classifier can decide whether a cropped carton image appears damaged.
It is simple to deploy but may rely on background or unrelated marks.
It cannot locate a defect without additional validated mechanisms.

A detector can propose boxes for products, gaps, labels, or damage regions.
Its box match depends on class, location, overlap rule, and duplicate handling.
Crowded shelves make object separation and occlusion especially difficult.
A model that detects objects well may still assign the wrong SKU.

Segmentation can outline a tear or a seal region.
It can be useful when defect extent matters,
but a pixel mask is not a calibrated physical measurement by itself.
A scale reference and geometric calibration are needed for millimeters or area.

OCR or a barcode reader can contribute identity or lot information.
The output should be checked against permitted formats and authoritative records.
Reading a code is not evidence that the content is authentic or undamaged.

Anomaly detection can learn from acceptable examples when defects are rare.
It detects deviation from the training distribution, not necessarily harm.
A harmless packaging redesign can trigger it,
while a familiar-looking dangerous defect may be missed.
A representative set of defect examples remains necessary for evaluation.

A vision-language model can combine image and text instructions,
for example to propose a description of visible carton damage.
It can omit a defect, misread a label, or invent an explanation.
A generated description is a proposed observation, not a quality disposition.
Restrict outputs to a schema and verify the visible evidence in the workflow.
Prompt changes, model updates, and image resizing all require change control.

A task-specific model trained on local labeled examples may outperform a
large general model, or the reverse; this is an empirical question.
Include a rules-based and manual baseline where practical.
Account for labeling, compute, hardware, review, and maintenance in the comparison.

### 5.7 Partition data before tuning

**Input:** eligible captured units, labels, metadata, and anticipated deployment.
**Output:** development, tuning, and untouched evaluation partitions.

The training set fits model parameters.
The validation set selects models, thresholds, prompts, and preprocessing.
The test set estimates performance after those decisions are frozen.
Test outcomes must not be repeatedly used to choose a preferred system.

Split at a level that prevents copies of the same underlying event or object
from appearing across development and test.
For video, keep all frames from a clip and all views of a package together.
For shelves, keep near-duplicate images from a shelf visit together.
If the intended deployment is new stores, hold out complete stores;
if the intended deployment is the future, hold out a later time period.
A combined site-and-time holdout is stronger when sufficient data exist.

Record related product lots, cameras, shifts, and capture sessions.
A carton seen before and after an image transformation is the same underlying unit.
A random frame split can leak packaging identity, background, and label cues.
The resulting score estimates recognition of familiar scenes,
not necessarily performance on unseen sites or future packaging.

Pretrained models may have seen public benchmark images.
Record model provenance where possible and avoid claiming independence
when training data or benchmark exposure cannot be established.
A private prospective local test with previously unseen physical units
is more relevant to the local deployment decision.

### 5.8 Sample to estimate real operations

**Input:** deployment population and frozen candidate.
**Output:** reference sample and an estimation plan with known inclusion rules.

Randomly sample from all eligible units, including no-alert units,
image-quality failures, and units for which the system abstained.
Otherwise false negatives and coverage are invisible.
A second sample may deliberately oversample rare defects or hard conditions
to learn about performance in those groups.
Keep its sampling probabilities and analyze it separately or weight it properly.

For a stratum such as site or lighting condition, a weighted count can use
weight = population units in the stratum / sampled units in the stratum,
when sampling is uniform within that stratum.
Sum weighted TP, FP, FN, and TN across strata before calculating population rates.
This requires known sampling and representative sampling within each stratum.
If inclusion probabilities are unknown, do not call the result population precision.

An enriched test with equal positive and negative counts can help compare recall
but its raw precision is not expected precision in a low-prevalence operation.
Report prevalence, stratum counts, exclusions, and confidence intervals.
If rare severe defects are absent from a sample, the data cannot establish
adequate sensitivity to them regardless of a high overall accuracy.

A sample of consecutive frames is not independent when frames share a carton.
Uncertainty calculations should resample or analyze independent clusters,
such as site visits or packages, as appropriate to the deployment question.
If there are only a few sites, a narrow frame-level interval is misleading.

### 5.9 Compute metrics at the action unit

**Input:** linked model decisions and independently established labels.
**Output:** confusion matrix, uncertainty, subgroup results, and queue estimate.

For a binary alert task, classify every eligible resolved unit as TP, FP, FN, or TN.
TP and FN sum to the actual positive units.
FP and TN sum to the actual negative units.
Precision = TP / (TP + FP), if at least one alert exists.
Recall = TP / (TP + FN), if at least one actual positive exists.
Specificity = TN / (TN + FP), if at least one actual negative exists.
FPR = FP / (FP + TN), if at least one actual negative exists.
Accuracy = (TP + TN) / all resolved units.

Accuracy can be misleading for rare defects.
If 99 of 100 cartons are sound, always declaring sound gives 99% accuracy
but detects none of the one defective carton.
A successful model needs both suitable detection and an actionable workload.

If recall and FPR are stable at the selected threshold,
expected precision at prevalence p is:

```text
precision = (recall * p) / (recall * p + FPR * (1 - p))
```

The formula assumes that the positive and negative units in deployment behave
like those used to estimate recall and FPR.
It is a scenario calculation, not proof that those rates will transfer.

For object detection, define how predicted boxes match reference boxes,
including the IoU threshold, class matching, and duplicate-prediction rule.
A model can find the correct region but misclassify the item.
Report localization and identity errors as well as downstream action outcomes.
For video, aggregate repeated predictions into an event using a frozen rule,
then measure event-level recall, precision, duplicate alerts, and latency.

### 5.10 Set thresholds for consequences and capacity

**Input:** tuning data, cost estimates, service capacity, and risk limits.
**Output:** threshold, abstention band, and escalation policy.

1. Generate a table of threshold versus TP, FP, FN, and review volume.
2. Estimate the consequence of each error for the actual decision.
3. Include time and capacity for reviewing alerts and low-quality images.
4. Respect any safety requirement that cannot be traded away for savings.
5. Choose a threshold on the tuning set, not the final test set.
6. Freeze the complete policy, including image quality gates and aggregation.
7. Evaluate the frozen policy on independent units and report uncertainty.

A lower threshold usually alerts on more cases and can improve recall,
but may increase false alerts and reduce precision.
That trend is not guaranteed at every discrete threshold in every finite sample.
Choose an operating point for the workflow, not a visually appealing curve.

A score is not automatically a probability.
A calibrated score of 0.8 would mean that approximately 80% of comparable
cases scored near 0.8 are positive in the relevant population.
Calibration can change with site, prevalence, lighting, or packaging.
If the score is not validated as a probability, call it a score.

An abstention band can protect ambiguous cases,
but the queue and the consequences of an unanswered case must be explicit.
A human fallback has a real service time and error rate.
Never hide abstentions in the negative class or exclude them from coverage.

### 5.11 Run a shadow evaluation and controlled pilot

**Input:** frozen policy, approved evaluation protocol, and quality owner.
**Output:** operational evidence and an explicit launch or stop decision.

During shadow mode, capture and score without changing operational decisions.
Independently establish reference outcomes for a sample of alerts and non-alerts.
Measure end-to-end latency, image coverage, identity joins, and workload.
Shadow mode cannot by itself prove that alerts cause better outcomes.

For a prospective pilot, compare a model-assisted workflow with the current
manual workflow under a fair allocation of sites, shifts, or time periods.
Where feasible, assign comparable units or sites randomly before deployment.
Record spillovers, site differences, operational changes, and noncompliance.
If assignment is not random, explain confounding and avoid a causal claim
stronger than the design permits.

Measure confirmed defects caught before dispatch, corrected shelf gaps,
time to action, units quarantined, false holds, and labor minutes.
For a shelf alert, verify both that a gap existed and that the response
was feasible; extra sales cannot simply be inferred from an empty facing.
For a carton, link the alert to an independent disposition and its timing.

Set minimum sample sizes and decision criteria before seeing pilot outcomes.
For rare but serious failures, define a separate safety assurance process.
A pilot may be underpowered to establish a very low severe-defect miss rate.
Do not reinterpret an absence of observed failures as proof of zero risk.

### 5.12 Operate, monitor, and roll back

**Input:** approved system, owners, baseline rates, and incident procedure.
**Output:** monitored service with retrievable evidence and a tested fallback.

1. Monitor eligible unit count, missing images, unreadable images, and abstentions.
2. Monitor capture latency, queue age, response time, and camera health.
3. Monitor alert rate and prediction mix by site, shift, SKU, and camera.
4. Audit a probability sample of non-alerts as well as alerts.
5. Monitor reviewer disagreement, override reasons, and resolved defect rates.
6. Investigate changes in packaging, lighting, shelves, cameras, and planograms.
7. Log model, threshold, preprocessing, prompt, and reference-policy versions.
8. Set explicit stop conditions and authorize a named owner to disable the model.
9. Revert to an operationally staffed manual process when controls fail.
10. Revalidate substantive changes on untouched data before promotion.

A shift in alert rate alone does not establish drift in model quality.
It may reflect a real change in defect prevalence or store activity.
Sampled ground truth is necessary to distinguish these explanations.
Delayed labels and selection into review can make apparent precision misleading.

A rollback plan includes who notices the fault, who has authority,
how the queue is drained, what happens to already-held units,
and how the manual process is staffed.
A switch with no operational fallback is not a complete rollback plan.

## 6. Worked example

### 6.1 Business decision and scope

Northstar Distribution is a fictional distribution center.
It receives sealed cartons of a shelf-stable consumer product.
The quality manager wants to identify visibly torn cartons before put-away.
The visual model proposes a review; a trained inspector decides disposition.
It does not automatically reject a supplier or establish a carrier claim.

The inspection unit is one carton presented at the receiving camera.
The defined positive is a visible tear exposing the inner packaging
on the camera-facing surfaces, confirmed by an independent inspector.
Other scuffs are negative for this particular visual task.
Cartons with an unreadable image or unobserved required surface are routed
to the existing manual process and counted separately as coverage failures.
The design does not claim detection of hidden damage or product contamination.

For this deliberately simplified arithmetic case, assume all 10,000 eligible
test cartons have usable views and resolved independent labels.
Actual deployments must report missingness and evaluate the fallback.
The data represent a future period and previously unseen cartons.
The threshold and event matching rule were frozen before the test.

### 6.2 Reference results

The reference review finds 400 positive cartons and 9,600 negative cartons.
The deployed policy alerts on 360 positives and 480 negatives.
The remaining 40 positives and 9,120 negatives do not generate an alert.
Each carton appears once in the table even if the camera records multiple frames.

| Frozen policy | Positive reference | Negative reference | Total cartons |
|---|---:|---:|---:|
| Alert | 360 | 480 | 840 |
| No alert | 40 | 9,120 | 9,160 |
| Total | 400 | 9,600 | 10,000 |

Prevalence = 400 / 10,000 = 0.04, or 4% of cartons.
Recall = 360 / 400 = 0.90, or 90% of positive cartons.
Specificity = 9,120 / 9,600 = 0.95, or 95% of negative cartons.
FPR = 480 / 9,600 = 0.05, or 5% of negative cartons.
Precision = 360 / 840 = 0.4286, or approximately 42.86% of alerts.
Accuracy = (360 + 9,120) / 10,000 = 94.8% of cartons.

The 94.8% accuracy figure is not sufficient for the launch decision.
The policy misses 40 positive cartons and sends 480 sound cartons to review.
A queue manager must plan for all 840 alerts, not just the 360 true alerts.
The example assumes every alert is acted on in time;
a real pilot must measure missed or late responses and reviewer errors.

### 6.3 Expected operating workload

Assume 250,000 eligible cartons per year at 4% prevalence,
90% recall, and 5% FPR.
These are scenario assumptions, not claims that the test rates are certain
or remain stable over sites and time.
There are 10,000 positive cartons and 240,000 negative cartons per year.
Expected TP = 10,000 × 0.90 = 9,000 cartons per year.
Expected FN = 10,000 × 0.10 = 1,000 cartons per year.
Expected FP = 240,000 × 0.05 = 12,000 cartons per year.
Expected TN = 240,000 × 0.95 = 228,000 cartons per year.
Expected alert workload = 9,000 + 12,000 = 21,000 cartons per year.

The implied precision is 9,000 / 21,000 = 42.86%.
At 2 minutes per alert, the review workload is 42,000 minutes per year,
or 700 hours per year before breaks, travel, and queue management.
At 250 operating days per year, the average is 84 alerts per day.
Daily volume can be uneven, so average staffing is not a peak-capacity plan.

If prevalence instead falls to 1% while recall and FPR stay the same,
positive cartons are 2,500 per year and negative cartons are 247,500 per year.
Expected TP is 2,250 and FP is 12,375 cartons per year.
Precision becomes 2,250 / (2,250 + 12,375) = approximately 15.38%.
This illustrates the base-rate effect; unchanged recall and FPR are assumptions.
The 1% scenario is not automatically the operation's actual future prevalence.

### 6.4 Compare against the actual manual baseline

Suppose the current process visually checks 100% of cartons.
Its independent measured recall for this defined tear is 80%.
Its initial check costs 0.30 minutes per carton at a loaded labor rate
of USD 24 per hour, or USD 0.40 per carton.
Confirmed and false alerts under either process receive an additional
2-minute secondary review at the same USD 24 per hour rate.
Assume secondary review correctly resolves all alerted cartons,
is completed before put-away, and is not included in the initial check time.

For the manual baseline, assume an observed 4% FPR.
Expected baseline TP = 10,000 × 0.80 = 8,000 cartons per year.
Expected baseline FN = 2,000 cartons per year.
Expected baseline FP = 240,000 × 0.04 = 9,600 cartons per year.
Expected baseline alerts = 17,600 cartons per year.
Baseline initial checking cost = 250,000 × USD 0.40 = USD 100,000 per year.
Baseline secondary review cost = 17,600 × (2 / 60) × USD 24
= USD 14,080 per year.
Baseline total inspection labor = USD 114,080 per year.

The proposed camera workflow replaces the initial manual check,
but retains secondary human review for every alert.
Camera-assisted review labor = 21,000 × (2 / 60) × USD 24
= USD 16,800 per year.
Assume an annual camera and software service cost of USD 45,000,
plus USD 12,000 per year for audit, upkeep, and calibration.
The proposed process therefore costs USD 73,800 per year
before the expected cost of missed damage.
The initial installation cost is discussed separately below.

Assume each missed positive causes USD 60 of expected incremental downstream
loss, including only consequences not already included in inspection labor.
That is a probability-weighted scenario value, not the price of every tear.
Baseline missed-defect loss = 2,000 × USD 60 = USD 120,000 per year.
Proposed missed-defect loss = 1,000 × USD 60 = USD 60,000 per year.
Baseline total modeled annual cost = USD 114,080 + USD 120,000
= USD 234,080 per year.
Proposed total modeled annual cost = USD 73,800 + USD 60,000
= USD 133,800 per year.
Modeled annual incremental benefit = USD 234,080 − USD 133,800
= USD 100,280 per year.

This calculation compares two complete workflows under stated assumptions.
It does not count the gross value of all 9,000 detected tears as incremental:
the manual baseline would already find 8,000 of them on average.
It also does not assume a labor saving is cash-releasable automatically.
The manager must establish whether hours translate to avoided overtime,
redeployment, or actual expenditure reduction.

### 6.5 Investment, sensitivity, and decision

Suppose the one-time installation and integration cost is USD 180,000.
With constant annual benefit and no discounting, simple payback is
USD 180,000 / USD 100,280 ≈ 1.80 years.
This is not net present value and ignores taxes, working capital,
equipment replacement, and the time value of money.
A capital approval requires the organization's actual investment method.

The savings before missed-defect loss are USD 114,080 − USD 73,800
= USD 40,280 per year.
The model prevents an expected 1,000 additional misses per year.
At an assumed loss of USD 20 rather than USD 60 per missed carton,
modeled annual benefit is USD 40,280 + 1,000 × USD 20
= USD 60,280 per year, before installation.
At USD 0 loss per miss the annual modeled benefit is USD 40,280,
assuming all other costs and performance remain unchanged.

If assisted recall falls to the manual baseline's 80% with the same FPR,
there is no modeled improvement in missed-defect loss;
the annual modeled benefit is USD 40,280 before installation.
If false alerts rise, additional review cost and delays must be added.
If fallback or exception volume is material, include its cost and residual misses.
These one-variable sensitivities do not represent joint worst-case conditions.

A reasonable decision is to authorize a bounded prospective pilot,
not an immediate permanent cutover.
The owner should require representative event-level results,
measured exception volume, workload by shift, independent reference checks,
privacy approval, and a functioning fallback.
If safety policy requires near-perfect detection of a severe defect,
the illustrative 90% recall alone does not establish acceptance.

## 7. Code, query, or calculation

The following Python 3 script uses only the standard library.
It computes the case confusion matrix, annual workload,
comparison with the manual baseline, and a prevalence sensitivity.
Save it as `inspection_economics.py` and run `python inspection_economics.py`.
It is a deterministic calculation, not image-processing or model-training code.
All values are fictional assumptions and counts may be fractional in a forecast.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Rates:
    prevalence: float
    recall: float
    false_positive_rate: float


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(tp, fp, fn, tn):
    return {
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "accuracy": safe_ratio(tp + tn, tp + fp + fn + tn),
    }


def expected_counts(volume, rates):
    positives = volume * rates.prevalence
    negatives = volume - positives
    tp = positives * rates.recall
    fn = positives - tp
    fp = negatives * rates.false_positive_rate
    tn = negatives - fp
    return tp, fp, fn, tn


def annual_cost(counts, initial_check_cost, review_minutes,
                hourly_wage, fixed_cost, loss_per_miss):
    tp, fp, fn, _ = counts
    review_cost = (tp + fp) * review_minutes / 60 * hourly_wage
    miss_cost = fn * loss_per_miss
    return initial_check_cost + review_cost + fixed_cost + miss_cost


def show_metrics(label, counts):
    tp, fp, fn, tn = counts
    result = metrics(tp, fp, fn, tn)
    print(f"{label}: TP={tp:,.0f}, FP={fp:,.0f}, "
          f"FN={fn:,.0f}, TN={tn:,.0f}")
    print("  precision={:.2%}, recall={:.2%}, "
          "specificity={:.2%}, accuracy={:.2%}".format(
              result["precision"], result["recall"],
              result["specificity"], result["accuracy"]))


test_counts = (360, 480, 40, 9120)
show_metrics("Test cartons", test_counts)

volume = 250_000
model = expected_counts(volume, Rates(0.04, 0.90, 0.05))
manual = expected_counts(volume, Rates(0.04, 0.80, 0.04))
show_metrics("Model annual", model)
show_metrics("Manual annual", manual)

model_cost = annual_cost(
    model, initial_check_cost=0, review_minutes=2,
    hourly_wage=24, fixed_cost=45_000 + 12_000, loss_per_miss=60,
)
manual_cost = annual_cost(
    manual, initial_check_cost=volume * 0.40, review_minutes=2,
    hourly_wage=24, fixed_cost=0, loss_per_miss=60,
)
benefit = manual_cost - model_cost
payback = 180_000 / benefit if benefit > 0 else None
print(f"Annual manual cost: USD {manual_cost:,.0f}")
print(f"Annual model cost: USD {model_cost:,.0f}")
print(f"Annual benefit: USD {benefit:,.0f}")
print(f"Simple payback: {payback:.2f} years" if payback is not None
      else "No simple payback at this annual benefit")

low_prevalence = expected_counts(volume, Rates(0.01, 0.90, 0.05))
show_metrics("Model at 1% prevalence", low_prevalence)
```

Expected output, with percentages rounded to two decimal places:

```text
Test cartons: TP=360, FP=480, FN=40, TN=9,120
  precision=42.86%, recall=90.00%, specificity=95.00%, accuracy=94.80%
Model annual: TP=9,000, FP=12,000, FN=1,000, TN=228,000
  precision=42.86%, recall=90.00%, specificity=95.00%, accuracy=94.80%
Manual annual: TP=8,000, FP=9,600, FN=2,000, TN=230,400
  precision=45.45%, recall=80.00%, specificity=96.00%, accuracy=95.36%
Annual manual cost: USD 234,080
Annual model cost: USD 133,800
Annual benefit: USD 100,280
Simple payback: 1.80 years
Model at 1% prevalence: TP=2,250, FP=12,375, FN=250, TN=235,125
  precision=15.38%, recall=90.00%, specificity=95.00%, accuracy=94.95%
```

The script assumes a fixed volume, rate stability, and full review coverage.
It does not model confidence intervals, peak queues, delayed interventions,
partial camera coverage, correlation among cartons, or discounting.
When a denominator is zero, `safe_ratio` returns `None`;
this demonstration's formatted metric output assumes nonzero denominators.
A production analysis should explicitly report undefined metrics rather than
formatting them as percentages and should validate its input rates.

To adapt this calculation, replace inputs with measured rates from an
independent evaluation and document the population and date of each estimate.
Keep the baseline and assisted workflow boundaries consistent.
Separate one-time costs from recurring annual costs and specify the currency.
Do not use this simple arithmetic as a substitute for safety validation.

## 8. Data, standards, and regulatory context

### 8.1 A minimum evidence record

A production record needs a unit ID and a link to source evidence.
It also needs capture time, local time zone, camera identity, site,
view or coverage status, image-quality outcome, and the task version.
For a shelf, record the applicable location and planogram version.
For a carton, record the verified shipment and package identity where available.
For video, record the clip window, event boundary, and deduplication rule.

Keep the source image or approved derivative separate from the prediction.
The prediction record should name model, preprocessing, threshold,
output, confidence or score semantics, and inference time.
A reviewer record should name the disposition, reason, and time;
a subsequent operational action should have its own authority and timestamp.
Record corrections as new events so that historical analyses remain reproducible.

When images are removed under retention rules, retain only permitted metadata.
An evidence pointer does not override a deletion obligation.
Define what remains auditable after deletion before deployment.
Do not retain person-identifiable frames indefinitely for convenience.

### 8.2 Identifiers and interoperability

GS1 publishes standards for product identification and barcode representation.
A Global Trade Item Number (GTIN) can identify a trade item
when correctly encoded, scanned, and mapped in the local system.
It does not automatically identify a unique physical carton or confirm quality.
A serial shipping container code (SSCC) can identify a logistics unit
when it is correctly assigned and linked to the relevant logistics records.

A camera may see a barcode that cannot be decoded because of damage or angle.
A successful decode may still refer to a stale or incorrectly applied label.
Track scan confidence, mapping validity, and unresolved identity separately.
Local identifiers should map to external identifiers with effective dates.
These standards are conventions unless a contract or applicable rule requires them.

A planogram is an operational source, not a universal truth standard.
Store changes, substitutions, and local exceptions must be versioned.
Document whether a shelf position is identified by physical coordinates,
a business location code, or an image-derived region.
Do not join images to the latest planogram when analyzing a past capture.

### 8.3 Quality and safety controls

The International Organization for Standardization's ISO 9001 page describes
requirements for a quality management system; applicability depends on the
organization's scope, certification arrangements, and contractual obligations.
A vision score alone does not establish ISO 9001 compliance or product conformity.
A site quality owner must define approved inspection and disposition procedures.

In regulated food, pharmaceutical, medical-device, or other safety-sensitive
contexts, specific laws, standards, and validation requirements may apply.
The controlling requirements depend on product, market, intended use,
and the action delegated to the system.
This educational material is not a legal or quality-release determination.
Consult qualified regulatory and quality personnel before operational use.

A vision system may supplement a required test, but cannot silently replace it.
Keep physical reference methods and calibration records where required.
Document change control for a camera, lighting, image crop, model, threshold,
review procedure, and business-rule change.
A model update that changes outputs can change the validated process.

### 8.4 Privacy and workplace context

A shelf image can include a customer's face, a worker's name badge,
a license plate, or other information relating to an identifiable person.
Even if the objective is product availability, collection and subsequent joins
may constitute processing of personal data under applicable privacy law.
A camera installation also may be subject to local workplace, surveillance,
consumer, biometric, or labor rules and collective agreements.
Applicability depends on jurisdiction and circumstances.

The European Union General Data Protection Regulation (GDPR), Regulation
(EU) 2016/679, sets conditions for processing personal data and includes
principles such as purpose limitation and data minimization.
It is binding where applicable; it does not prescribe a universal retention period
or make every workplace camera lawful merely because a sign is posted.
An appropriate lawful basis and any required notices, assessments,
rights processes, and safeguards require jurisdiction-specific review.

The EU AI Act, Regulation (EU) 2024/1689, has phased application provisions.
As of 2026, applicability and obligations depend on the system, role,
use case, and applicable dates; do not assume that every inspection system
is high-risk or that a vendor's label settles the classification.
Review current authoritative text and qualified counsel's assessment.

### 8.5 Practical privacy and security design

1. Restrict the field of view to the shelf or inspection zone when feasible.
2. Avoid audio and unrelated recording when they are not needed for the purpose.
3. Assess whether edge processing can avoid transferring raw full-frame video.
4. Mask or crop people when appropriate, and test whether masking works.
5. Treat reversible blur or imperfect masking as potentially personal data.
6. Limit access by role and log viewing, export, and administrative changes.
7. Encrypt data in transit and at rest under the organization's security policy.
8. Establish retention periods and delete derivatives and backups as applicable.
9. Prohibit secondary worker-performance or disciplinary use unless separately
   justified, disclosed, and approved under applicable requirements.
10. Review vendor subprocessors, cross-border transfers, training use,
    incident notification, deletion, and return of data in the contract.

These controls reduce risk but do not alone establish legal compliance.
Access to a narrow crop can still expose sensitive product or site information.
A vendor may cache images or send them to another service unless terms and
technical controls prevent that behavior.
Verify the actual processing path and administrative permissions.

### 8.6 AI governance and evidence

NIST's AI Risk Management Framework 1.0 is a voluntary framework,
not a universal legal certification for a vision model.
Its govern, map, measure, and manage functions can organize owners,
intended use, evaluation, incident response, and monitoring.
A model card or supplier statement is evidence to assess, not proof of fitness.

Maintain a record of training-data provenance where available,
rights to use images, model version, evaluation population,
known limitations, decisions on unacceptable risk, and approval authority.
A supplier should disclose when weights, prompts, data routing,
or model-serving regions change, subject to contract and technical capability.
A system cannot be considered stable merely because its API name is unchanged.

## 9. Industry practice and case evidence

### 9.1 Walmart shelf-scanning robots

In November 2020, Associated Press reported that Walmart was ending its work
with shelf-scanning robots that had been deployed in approximately 500 stores.
The report described the company turning to employees for shelf checks.
This is press reporting about a specific historical deployment,
not a controlled trial of all shelf-image systems or current Walmart practice.
See the source in section 16 for the article and date.

The case illustrates that a technically possible shelf scan must compete with
existing store workflows, including people already moving through the aisles.
[INFERENCE] Robot acquisition, aisle navigation, maintenance,
association with planogram positions, and alert response can all affect value.
The cited report does not quantify each of those costs or establish the model's
precision, recall, or a causal explanation for every implementation decision.

For a new retailer, the decision should therefore compare a complete service:
capture, mapping, integration, exception handling, and store labor.
An image-only accuracy number is not enough to forecast net value.
The case does not justify assuming that another camera architecture will fail.

### 9.2 BMW visual inspection

BMW Group described production uses of artificial intelligence and image
recognition in an official 2019 press release.
Its examples include checking component features and comparing images
with reference images in automotive manufacturing.
This is a company account of its own applications, not an independent audit
or evidence that a particular model is safe in a different factory.

[INFERENCE] A controlled factory station may offer more stable geometry
and lighting than a public retail aisle, but site-specific tests are still needed.
The company account does not provide the event-level confusion matrix,
confidence intervals, total integration cost, or counterfactual economics
needed to calculate the Northstar case.
Do not transfer its deployment description into a performance guarantee.

### 9.3 Public shelf detection research

Goldman and coauthors' 2019 CVPR paper, *Precise Detection in Densely
Packed Scenes*, introduces the SKU-110K retail shelf detection dataset.
The paper is useful evidence that locating products in crowded retail images
is a studied technical problem and that occlusion and density matter.
It is not an audit of a particular retailer's shelf availability workflow.

The dataset task and label definitions must be compared to the business task.
Detecting product objects in an image does not automatically determine
which SKU belongs in an empty location or whether a store has back-room stock.
A public image score cannot establish local privacy compliance,
response times, or incremental store revenue.
Public benchmark images may also be familiar to pretrained systems.

### 9.4 Public industrial anomaly research

Bergmann and coauthors' MVTec AD work presents an industrial anomaly
benchmark with object and texture categories and defect annotations.
It is a useful way to study detection and localization of visual anomalies.
The benchmark is not a substitute for evaluating a specific site's materials,
illumination, defect severity, throughput, and quality-release process.

Anomaly scores can highlight unusual texture, contamination, or deformation.
They cannot by themselves decide whether a deviation is economically material
or whether a package's unseen contents are acceptable.
The site's reference standard and error consequences remain decisive.

### 9.5 How to use case evidence

Separate what a source reports from what you infer for a new deployment.
For company statements, ask what was measured, by whom, and against what baseline.
For academic benchmarks, ask how examples were collected and split.
For a press report, distinguish an observed business decision from its causes.
For any source, look for dates and whether the described system still operates.
None of these sources supplies the fictional case's rates or costs.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Blocked shelf view | Many uncertain regions near shoppers or carts | Occlusion hides the assigned facing | False gap alerts or missed gaps | Mark unknown; recapture or route to an associate |
| Stale planogram | Alert clusters after a reset | Old assignment is joined to new shelf layout | Work on the wrong item | Effective-date joins and reset approval checks |
| Lookalike packages | Errors concentrated in similar colors or sizes | Visual identity lacks distinguishing evidence | Wrong SKU replenishment | Verify barcode, shelf label, and identity confidence |
| Hidden carton damage | Complaints on surfaces never imaged | Camera sees only one side | Unsafe acceptance or missed claim evidence | Multiple views or manual inspection for unobserved surfaces |
| Glare or motion blur | Image-quality failures rise on one shift | Lighting or belt speed changes | Small tears disappear | Measure image quality and use a safe fallback |
| Random frame split | Extremely high test score on familiar backgrounds | Adjacent frames leak between sets | Inflated generalization estimate | Group by physical event and hold out sites or time |
| Alert-only labels | Reported precision without a non-alert audit | Unreviewed misses have no labels | Unknown recall and undetected harm | Sample all eligible units including non-alerts |
| Enriched-test precision | Balanced benchmark produces high precision | Deployment has lower prevalence | Review workload is underestimated | Estimate using representative data or known sampling weights |
| Duplicate video tracks | More alerts than physical cartons | Occlusion fragments tracks | Duplicate handling and inflated counts | Event-level deduplication with audited IDs |
| Merged video tracks | One track covers several overlapping cartons | Tracker loses object identity | Multiple missed or misassigned inspections | Controlled spacing, additional view, and exception route |
| Low-confidence forced decision | Unknown cases appear as negatives | System lacks an abstention state | Hidden coverage failure | Preserve unknown and monitor manual fallback |
| Score treated as probability | A 0.9 score is described as 90% certainty without evidence | Scores are not calibrated locally | Incorrect risk decisions | Test calibration by relevant population and time |
| Rare severe defect omitted | No severe cases in evaluation | Convenience sample lacks the important class | False assurance of safety | Purposeful challenge testing and safety assurance |
| Reviewer anchoring | Reviewers almost always accept model proposals | Display of a confident label biases judgment | Errors become reference labels | Independent blinded audits and reasoned overrides |
| Queue overload | Alert age rises above the decision deadline | Average capacity ignores peaks | True alerts arrive too late | Measure peaks, cap workload safely, staff fallback |
| Wrong physical join | Carton ID differs from scanner or manifest | Clock drift or ambiguous mapping | Action applied to another unit | Verify keys, timestamps, and conflicting joins |
| Generated explanation invented | Text describes an unseen puncture or cause | Vision-language model fills in plausible detail | Unsupported claim or disposition | Require visible evidence and human verification |
| Packaging redesign | One SKU's alerts spike after a supplier change | Appearance shifts from training data | False alarms or missed defects | Change notification and validation before promotion |
| Camera or lighting change | Quality changes after maintenance | Crop, angle, and illumination shift | Model performance no longer applies | Version hardware and revalidate after changes |
| Privacy scope creep | Footage is requested for unrelated staff monitoring | Repurposing exceeds approved purpose | Loss of trust or legal exposure | Purpose limits, access governance, and review |
| Vendor silent update | Scores change without a local release | Provider changes model or preprocessing | Uncontrolled behavior and audit gap | Version pinning, notification, contract, and canary tests |
| Missing fallback | Camera offline while receiving continues | Manual procedure is not staffed | Uninspected units proceed | Tested rollback, monitoring, and assigned ownership |
| Misleading safety claim | External appearance used as evidence of internal integrity | Visual task is narrower than quality outcome | Harmful release decision | Retain appropriate physical tests and quality authority |

A red flag is a reason to investigate, not proof of the stated mechanism.
Record the incident, affected time window, units, decision impact, and owner.
If the approved risk boundary is exceeded, stop the assisted workflow
and follow the documented manual or quality-hold procedure.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage it |
|---|---|---|---|
| Target is visually observable | Image evidence must contain the relevant signal | Internal defect or hidden surface | Inspect views and compare with physical reference tests |
| Relevant surfaces are covered | A negative decision needs adequate visibility | Occlusion, underside, glare, or missing frame | Coverage audit and explicit unknown state |
| Reference labels are sufficiently reliable | Metrics require a defensible comparison | Ambiguous labels or model-influenced reviewers | Independent adjudication and label agreement studies |
| Eligible population is known | Rates must have a meaningful denominator | Lost camera events or excluded offline periods | Reconcile to independent event or scanner counts |
| Sampling is representative or weighted | Test results must relate to deployment | Convenience selection or unrecorded oversampling | Record inclusion probabilities and report stratum counts |
| Test units are independent enough for analysis | Uncertainty must not be artificially narrow | Adjacent frames or repeated visits treated as independent | Group by carton, session, site, or relevant cluster |
| Recall transfers to future operations | Scenario forecasts use the measured rate | New packaging, lighting, cameras, or site mix | Prospective holdout and ongoing non-alert audits |
| FPR transfers to future operations | Queue and precision depend on negative cases | New backgrounds, promotions, or prevalence-related mix shifts | Monitor by site and condition with fresh labels |
| Stable identity join | Correct action requires the correct physical unit | Clock drift, barcode collision, or tracking failure | Reconcile IDs and audit unmatched or conflicting records |
| Review occurs before deadline | Alerts help only if acted on in time | Overloaded queues or shift handoff | Measure end-to-end response and late-action rate |
| Secondary inspection resolves alerts | Economics assumes review distinguishes true and false alerts | Human error or inadequate procedure | Blind reinspection and measured reviewer agreement |
| Manual baseline is comparable | Incremental value requires a fair counterfactual | Different site mix, staffing, or observation periods | Concurrent comparison or justified adjustment |
| Cost estimates are incremental | Avoid counting work or loss twice | Shared costs, non-releasable labor, overlapping loss categories | Finance review and explicit cost boundary |
| Privacy controls cover actual processing | Data handling must match the approved purpose | Vendor cache, unauthorized export, or wider camera view | Data-flow review, access logs, and contract audit |
| Low-frequency serious outcomes are observable | Evidence must support the claimed risk level | Too few severe events in a pilot | Larger or targeted challenge studies and expert safety review |
| Rollback remains operational | Failed automation must not leave an inspection gap | Manual staffing removed or runbook untested | Exercise fallback and maintain a named owner |

The fictional cost case assumes 100% usable views for clarity.
It also assumes no model errors in the secondary human review.
Those simplifying assumptions are not recommendations for real deployment.
Sensitivity analysis should vary rates, labor, loss, coverage, and timing jointly
when the data and decision warrant a richer model.

A confidence interval expresses uncertainty under its sampling assumptions;
it does not correct biased labels, leakage, or a missing population segment.
For clustered data, the effective independent sample size may be far below
the number of frames.
A small number of sites limits claims about new sites even when each has
many thousands of images.

## 12. Exercises

### Exercise 1 — Separate evidence from action

A store photograph shows an apparently empty region near a cereal shelf label.
The planogram was last updated two weeks ago.
The stock ledger reports 12 units on hand, last refreshed four hours ago.
A vision-language model says, "The cereal is out of stock because the supplier
missed its delivery; reorder 24 units immediately."

**Deliverable:** A short decision note with four columns:
visible evidence, unverified inference, additional check, and authorized action.
**Acceptance criteria:** Distinguish a visual gap from availability,
identify the unsupported delivery-cause claim, verify planogram and stock,
and do not equate a model suggestion with a purchase commitment.
**Time:** 10 minutes.

### Exercise 2 — Specify three inspection tasks

Design one task each for shelf availability, torn cartons, and a missing seal.
Assume each task has a different operational owner and action deadline.
For the seal, external appearance is not a substitute for a leak test.

**Deliverable:** A one-page task and data contract for each case.
Include physical decision unit, positive label, unknown state,
required views, context keys, model family, human action, and fallback.
**Acceptance criteria:** A visible target is separated from hidden condition;
identity and capture time are preserved; unknown is not silently negative;
and the selected method is justified against a simpler alternative.
**Time:** 25 minutes.

### Exercise 3 — Capture and annotation design

A conveyor carries cartons at 600 mm per second.
The camera views 800 mm across 1,600 usable horizontal pixels.
At a 1/300-second exposure, estimate displacement during exposure.
A quality owner asks for visible tears as small as 3 mm.
The current camera only sees the top of each carton.

**Deliverable:** The two calculations and an annotation and capture plan.
**Acceptance criteria:** Compute mm per pixel and motion displacement in mm
and pixels; explain why the theoretical scale is not a detection guarantee;
include the unobserved sides, unknown labels, adjudication, and a fallback.
**Time:** 20 minutes.

### Exercise 4 — Calculate event-level metrics

On an independently labeled sample of 5,000 cartons,
a frozen alert policy produces 144 TP, 194 FP, 36 FN, and 4,626 TN.
Every row is one unique carton and all reference labels are resolved.

**Deliverable:** Compute prevalence, recall, specificity, FPR, precision,
accuracy, and total alert workload; interpret each denominator.
**Acceptance criteria:** Show the arithmetic and explain why high accuracy
is not sufficient and why test prevalence must be reported.
**Time:** 15 minutes.

### Exercise 5 — Compare workflows economically

A site handles 100,000 cartons per year with 2% positive prevalence.
The existing manual workflow costs USD 45,000 per year in inspection labor,
misses 400 positive cartons, and has no other modeled recurring cost.
The assisted workflow misses 200 positive cartons,
costs USD 24,000 per year for service and upkeep,
and costs USD 12,000 per year in review labor.
Assume each missed carton has USD 50 expected incremental loss.
Installation is USD 90,000 once; all amounts exclude discounting and tax.

**Deliverable:** Calculate the annual cost of each workflow,
the annual incremental benefit, and undiscounted simple payback.
State two assumptions that could change the answer.
**Acceptance criteria:** Do not count gross detected value as incremental;
include missed-defect loss in both workflows and installation only in payback.
**Time:** 15 minutes.

### Exercise 6 — Design a leakage-resistant pilot

A vendor offers 200,000 video frames from 400 cartons in four facilities.
Each carton appears in multiple adjacent frames.
One facility uses a new reflective wrapper and another has different lighting.
A local team proposes a random 80/20 frame split and will label only alerts.

**Deliverable:** A one-page alternative evaluation and pilot protocol.
**Acceptance criteria:** Group frames by physical carton and capture session;
justify site/time holdouts; label a sample of non-alerts and exclusions;
freeze thresholds; report event-level workload and uncertainty;
and include a feasible baseline, subgroup analysis, and stop condition.
**Time:** 25 minutes.

### Exercise 7 — Privacy and worker safeguards

A proposed shelf camera captures an aisle, occasional customers,
and staff name badges, and uploads video to a third-party service.
The operations leader wants to use the same footage for staff productivity
scoring without changing the original procurement description.

**Deliverable:** A data-flow and control review with an approval decision.
**Acceptance criteria:** Identify personal data and the new purpose;
propose minimization, access, retention, vendor, rights and local labor review;
explain why blur or a notice alone is not proof of compliance;
and identify who must approve before changing the purpose.
**Time:** 20 minutes.

### Exercise 8 — Select a release decision

A seal-inspection model detects 90 of 100 visible missing-seal cases in a
held-out sample and produces 300 false alerts on 9,900 negative units.
Its camera does not observe the underside.
The quality requirement concerns leaks, including defects invisible in images.
The vendor proposes immediate automatic release of non-alert units.

**Deliverable:** A release recommendation and a short evidence rubric.
**Acceptance criteria:** Compute visible-defect recall and FPR;
reject the unsupported inference from appearance to leak integrity;
identify missing coverage, reference tests, and consequence of misses;
and describe the conditions for a limited assisted pilot if appropriate.
**Time:** 15 minutes.

### Exercise 9 — Interpret prevalence shift

At a fixed threshold, suppose recall is 80% and FPR is 2%.
Calculate expected precision for 10,000 cartons when prevalence is 10%,
and again when prevalence is 1%.
Assume both rates transfer unchanged for this scenario.

**Deliverable:** Two expected confusion matrices and a two-sentence interpretation.
**Acceptance criteria:** Compute each expected count and precision;
explain that prevalence changes precision and review burden,
and identify the stability assumption rather than asserting it as fact.
**Time:** 15 minutes.

## 13. Self-check answers

### Exercise 1

The photograph may show a gap at one visible location at capture time.
It does not establish that the correct cereal was assigned there,
that the whole shelf is empty, or that the store has no stock elsewhere.
The ledger's 12 units are recorded on-hand quantity at its refresh time,
not proof of physical availability or a current supplier delivery state.

Check the planogram version, shelf label, adjacent facings, stock location,
and the supplier or receiving record before deciding what happened.
An associate can check and replenish under the store's ordinary authority.
A buyer may separately decide whether to order after checking the policy.
The stated supplier cause and 24-unit order are unsupported proposals.

### Exercise 2 rubric

A strong answer states one physical unit and a deadline for each task.
It requires the shelf location and effective planogram for availability,
the visible surface and independently confirmed tear policy for cartons,
and the quality procedure and physical leak test for seal integrity.
It distinguishes an image classifier from a detector or segmentation model
based on whether the workflow needs a whole-image label or a defect location.

Award full credit only when unreadable, blocked, or unobserved regions
have an explicit route and are included in coverage reporting.
A model output is a proposal; the answer names the reviewer or authorized owner.
A simple manual, sensor, or barcode baseline is included for comparison.
A design that automatically releases unobserved or untested seals fails.

### Exercise 3

Scale = 800 mm / 1,600 pixels = 0.5 mm per pixel.
Exposure is 1/300 second, so displacement = 600 / 300 = 2 mm.
At 0.5 mm per pixel, that displacement is approximately 4 pixels.
A 3 mm tear spans about 6 pixels under ideal geometry,
but contrast, motion, focus, compression, and orientation can obscure it.

The camera cannot establish the condition of the carton's sides or underside.
Add views or route unobserved surfaces to an approved manual process.
Annotators should distinguish visible tear, no visible tear, and not observable;
independently adjudicate disagreements against the physical carton where possible.
Test lighting and focus at operational speed with representative tear samples.

### Exercise 4

Positive reference units = 144 + 36 = 180 cartons.
Negative reference units = 194 + 4,626 = 4,820 cartons.
Prevalence = 180 / 5,000 = 3.6% of cartons.
Recall = 144 / 180 = 80% of positive cartons.
Specificity = 4,626 / 4,820 ≈ 95.98% of negative cartons.
FPR = 194 / 4,820 ≈ 4.02% of negative cartons.
Precision = 144 / (144 + 194) ≈ 42.60% of alerts.
Accuracy = (144 + 4,626) / 5,000 = 95.4% of cartons.
Total alert workload = 144 + 194 = 338 cartons in the sample.

The 36 missed positives are operationally important despite high accuracy.
If prevalence changes, precision can change even with stable conditional rates.
Sampling and label quality determine whether the result applies to the operation.

### Exercise 5

Manual missed-defect loss = 400 × USD 50 = USD 20,000 per year.
Manual total = USD 45,000 + USD 20,000 = USD 65,000 per year.
Assisted missed-defect loss = 200 × USD 50 = USD 10,000 per year.
Assisted total = USD 24,000 + USD 12,000 + USD 10,000
= USD 46,000 per year.
Annual incremental benefit = USD 65,000 − USD 46,000
= USD 19,000 per year.
Simple payback = USD 90,000 / USD 19,000 ≈ 4.74 years.

The answer assumes comparable carton mix and stable misses in each process.
It also assumes review labor and service costs are complete and non-overlapping,
and that the USD 50 loss is incremental and applies equally to each miss.
Discounting, safety requirements, and cash-releasable labor may change approval.

### Exercise 6 rubric

A strong protocol groups every frame and repeated view of each carton together.
It holds out a relevant site and later period where data permit,
while reporting that four sites limit generalization.
It treats the reflective wrapper and different lighting as prespecified slices,
not as post hoc exclusions.

The reference sample covers alerts, no-alerts, unreadable images,
and cases without a reliable identity link.
Sampling probabilities are recorded; enriched samples are weighted appropriately.
The test policy, threshold, and event deduplication are fixed before evaluation.
Report carton-level sensitivity, precision, coverage, duplicates, and latency,
plus cluster-aware uncertainty and counts by site and condition.
A pilot includes the current manual comparator and a stop or rollback owner.

### Exercise 7 rubric

A strong answer traces capture, edge processing, vendor upload,
operator access, downstream exports, storage, deletion, and subprocessors.
It identifies faces and badges as potentially personal data.
It treats productivity scoring as a separate purpose requiring review,
not an automatic extension of shelf availability monitoring.

Recommend a narrower view, short justified retention, restricted roles,
access logs, secure transfer and storage, vendor contractual controls,
and assessment of applicable privacy and workplace obligations.
Require privacy, legal, security, labor or employee-representation,
and business-owner review as appropriate to the jurisdiction and organization.
Blurring can fail and notice alone does not establish a lawful basis.

### Exercise 8

Recall for the visible missing-seal label is 90 / 100 = 90%.
FPR on labeled negative units is 300 / 9,900 ≈ 3.03%.
Precision among the 390 alerts is 90 / 390 ≈ 23.08%.
Ten visible missing seals were missed in this sample.

The reference task is visible missing seals, not all leaks.
The underside is unobserved and some leaks are visually undetectable.
Therefore the stated sample does not support automatic release for leak integrity.
Retain the approved leak test and release procedure.
A limited assisted pilot may route visual alerts to review after appropriate
quality, safety, coverage, reference, and operational approvals.

### Exercise 9

At 10% prevalence, positives = 1,000 and negatives = 9,000 cartons.
TP = 800, FN = 200, FP = 180, and TN = 8,820 cartons.
Precision = 800 / (800 + 180) ≈ 81.63%.
The expected alert workload is 980 cartons.

At 1% prevalence, positives = 100 and negatives = 9,900 cartons.
TP = 80, FN = 20, FP = 198, and TN = 9,702 cartons.
Precision = 80 / (80 + 198) ≈ 28.78%.
The expected alert workload is 278 cartons.

Precision falls because false alerts arise from a much larger negative group.
The calculation assumes 80% recall and 2% FPR remain valid in both populations;
new lighting, product mix, or sites could invalidate that assumption.

## 14. Interview questions

### 1. What does a shelf photograph actually prove?

**Strong answer:** It provides time-stamped visual evidence about the visible
region; assignment, total stock, cause, and action need separate checks.
**Weak answer:** It proves a supplier stockout and authorizes replenishment.

### 2. How would you choose a decision unit for video?

**Strong answer:** Use a physical carton or defined event, group frames,
handle split and merged tracks, and evaluate at the action level.
**Weak answer:** Treat every frame as an independent inspected carton.

### 3. When would you choose a detector instead of a classifier?

**Strong answer:** A detector helps locate multiple objects or regions;
choose it when location matters and verify box matching and downstream value.
**Weak answer:** A detector is always better because it is more complex.

### 4. How can a high accuracy model be unusable?

**Strong answer:** Rare positives can be missed while negatives dominate;
inspect recall, precision, coverage, consequences, and workload.
**Weak answer:** Accuracy above 95% is sufficient in every quality setting.

### 5. Why does precision change when defect prevalence changes?

**Strong answer:** The positive and negative populations change in relative size;
even a stable FPR can generate many false alerts at low prevalence.
**Weak answer:** The threshold alone uniquely determines precision everywhere.

### 6. How would you avoid leakage in a video dataset?

**Strong answer:** Group frames by event and physical unit,
hold out relevant sites or future periods, and freeze tuning before final test.
**Weak answer:** Shuffle frames randomly because that maximizes sample size.

### 7. Can a camera certify that a carton is undamaged?

**Strong answer:** Only the visible surfaces and defined defect can be assessed;
coverage gaps and hidden condition require other evidence or inspection.
**Weak answer:** A negative model score proves the contents are safe.

### 8. How do you evaluate the economics?

**Strong answer:** Compare complete assisted and manual workflows,
including changes in missed loss, review burden, fixed costs, and uncertainty.
**Weak answer:** Multiply every detected defect by its gross value as new savings.

### 9. What makes a human review step meaningful?

**Strong answer:** Reviewer authority, sufficient evidence and time,
independent audits, recorded overrides, and a safe escalation path.
**Weak answer:** A person clicks accept on every proposed label.

### 10. What privacy question arises from shelf cameras?

**Strong answer:** The image may contain people, and use, retention,
vendor transfers, access, and workplace implications need context-specific review.
**Weak answer:** No privacy issue exists because the objective is inventory.

### 11. What triggers revalidation?

**Strong answer:** Relevant changes in camera geometry, lighting, packaging,
model or prompt, threshold, site, label policy, or downstream procedure.
**Weak answer:** Only a complete rewrite of the model requires review.

### 12. How do you respond to a rising alert rate?

**Strong answer:** Check prevalence, camera quality, site and product mix,
model changes, and independently sampled labels before diagnosing drift.
**Weak answer:** Immediately lower the sensitivity without examining misses.

## 15. Key terms

- [Abstention](#4-definitions-and-vocabulary): explicit unknown or deferred inference.
- [Annotation policy](#54-define-annotation-and-reference-evidence): documented labeling rules.
- [Anomaly detection](#56-choose-a-model-for-the-question): scoring deviation from normal.
- [Barcode](#82-identifiers-and-interoperability): machine-readable representation of an identifier.
- [Bounding box](#4-definitions-and-vocabulary): rectangular region for an object or defect.
- [Calibration](#510-set-thresholds-for-consequences-and-capacity): agreement of probabilities with observed rates.
- [Classification](#56-choose-a-model-for-the-question): assignment of a category to an input.
- [Coverage](#52-define-the-population-and-unit): portion of eligible units adequately observed.
- [Data leakage](#57-partition-data-before-tuning): development access to inappropriate evaluation information.
- [Detection](#56-choose-a-model-for-the-question): locating and classifying objects or regions.
- [Drift](#512-operate-monitor-and-roll-back): change in inputs, relationships, or outcomes over time.
- [Event](#52-define-the-population-and-unit): a defined operational occurrence receiving a decision.
- [False negative](#59-compute-metrics-at-the-action-unit): missed positive reference unit.
- [False positive](#59-compute-metrics-at-the-action-unit): alert on a negative reference unit.
- [Field of view](#53-design-capture-around-the-smallest-relevant-defect): physical area captured by a camera.
- [Ground sampling distance](#53-design-capture-around-the-smallest-relevant-defect): approximate physical scale per pixel.
- [Human-in-the-loop](#511-run-a-shadow-evaluation-and-controlled-pilot): specified human verification or decision step.
- [Intersection over union](#4-definitions-and-vocabulary): overlap ratio for regions or boxes.
- [Latency](#512-operate-monitor-and-roll-back): time from capture to actionable outcome.
- [Multimodal join](#55-link-context-without-inventing-identity): linking visual evidence to other records.
- [Occlusion](#53-design-capture-around-the-smallest-relevant-defect): blocked view of the target.
- [Optical character recognition](#56-choose-a-model-for-the-question): extraction of text from images.
- [Planogram](#3-the-core-mental-model): versioned intended shelf placement.
- [Precision](#59-compute-metrics-at-the-action-unit): share of alerts that are true positives.
- [Prevalence](#58-sample-to-estimate-real-operations): share of units with the target condition.
- [Recall](#59-compute-metrics-at-the-action-unit): share of positive units detected.
- [Reference standard](#54-define-annotation-and-reference-evidence): independent basis for labels.
- [Segmentation](#56-choose-a-model-for-the-question): pixel- or region-level classification.
- [Specificity](#59-compute-metrics-at-the-action-unit): share of negatives not alerted.
- [Threshold](#510-set-thresholds-for-consequences-and-capacity): score cutoff used by a decision rule.
- [Tracking](#52-define-the-population-and-unit): connecting detections across frames.
- [Vision-language model](#56-choose-a-model-for-the-question): model processing image and text together.

## 16. Sources

The sources below support general methods and the specifically attributed cases.
They do not validate the fictional numerical example or replace local review.
Access current versions and determine applicability before relying on them.

1. Goldman et al., [Precise Detection in Densely Packed Scenes](https://openaccess.thecvf.com/content_CVPR_2019/html/Goldman_Precise_Detection_in_Densely_Packed_Scenes_CVPR_2019_paper.html), CVPR 2019.
   Introduces the SKU-110K shelf-image detection benchmark.
2. Bergmann et al., [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html), CVPR 2019.
   Describes an industrial visual anomaly benchmark and associated task.
3. Associated Press, [Walmart abandons shelf-scanning robots, lets humans do work](https://apnews.com/article/robotics-22afe5552f69e9970b386cabf6b24f83), 2020.
   Reports a historical retailer decision; not a general causal evaluation.
4. BMW Group, [Fast, efficient, reliable: Artificial intelligence in BMW Group Production](https://www.press.bmwgroup.com/global/article/detail/T0298650EN/fast-efficient-reliable:-artificial-intelligence-in-bmw-group-production), 2019.
   Company account of production image-recognition applications.
5. NIST, [Artificial Intelligence Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1), 2023.
   Voluntary AI risk management framework.
6. European Union, [General Data Protection Regulation, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), 2016.
   Primary legal text; jurisdiction and applicability must be assessed.
7. European Union, [Artificial Intelligence Act, Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), 2024.
   Primary legal text with phased applicability; assess current requirements.
8. GS1, [GS1 identification keys](https://www.gs1.org/standards/id-keys).
   Overview of identification keys including trade items and logistics units.
9. ISO, [ISO 9001 quality management systems](https://www.iso.org/standard/62085.html).
   Quality-management standard overview, not a certification of a model.
10. UK Information Commissioner's Office, [Guidance on AI and data protection](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/).
    Regulatory guidance relevant to applicable UK data-protection assessments.

## 17. Links

- [Module 18 topic index](_index.md).
- [Module 18 survey](../../../course/18_deep_learning_foundation_models_llm_agents.md).
- [Neural network fundamentals](01_neural-network-fundamentals.md).
- [Evaluating zero-shot and fine-tuned models](06_zero-shot-evaluation.md).
- [Benchmark contamination and leakage](07_benchmark-contamination.md).
- [LLMs for extraction and supplier communication](08_llm-extraction.md).
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
  Use the lab context to examine operating capacity and downstream decisions.
