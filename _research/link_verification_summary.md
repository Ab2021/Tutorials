# Web Source Verification Summary

**Verification date:** 24 October 2026
**Scope:** All external URLs cited in the course package, including the two research briefs and the 30 modules.
**Method:** Automated `GET` requests with a desktop user agent, 15-second timeout, redirects followed. Manual browser/search verification for dead or ambiguous links. DOI metadata checked through the Crossref REST API.
**Raw results:** `_research/link_check.tsv` (first pass, 324 unique URLs), `_research/link_check_post_fix.tsv` (post-fix, 375), `_research/link_check_final.tsv` (394, including historical URLs), and `_research/link_check_final2.tsv` (385 current user-facing URLs, excluding the audit report and this summary).

## 1. Post-fix results

| Status | Count | Interpretation |
|---|---:|---|
| 200 | 267 | Reachable |
| 403 | 83 | Usually bot-blocked, paywalled, or requires login; not evidence of a dead citation |
| 404 | 16 | All FDA/FTC anti-bot responses in the current set; genuine FDA moves were identified manually and corrected below |
| ERR | 8 | Timeout, TLS, or connection failure at HCUP, IMDRF, ML4H, Sleep Data, CIL Health, and PhysioNet |
| 202 | 9 | Accepted for processing; usually API/redirect behavior |
| 307 | 1 | Redirect |
| 203 | 1 | Non-authoritative information; usually a proxy/CDN response |

**Conclusion:** after replacing genuinely moved, superseded, or stale URLs, no cited source is known to be a confirmed dead 404. The sources added in Modules 14B and 23–26 (arXiv papers, ONNX/TensorRT/TFLite/CMSIS-NN/OpenVINO/ExecuTorch/llama.cpp/MLC-LLM, NVIDIA Jetson Thor, Arm Ethos-U85, Hailo, Coral) were separately checked; all resolved except bot-blocked FDA/IEC/ISO pages. The remaining 404s are FDA/FTC anti-bot responses, and the remaining errors are network/TLS failures at otherwise valid official sites. All should be rechecked in a browser before formal publication. A 404 from FDA is not automatically bot-blocking: four FDA pages in the first pass were genuinely moved.

## 2. Corrections applied

| Original | Replacement | Reason |
|---|---|---|
| `https://lincsproject.org/` | `https://clue.io/lincs` | Root 404; CLUE is the current LINCS L1000 access point |
| `https://midogpathology.grand-challenge.org/` | `https://midog2025.deepmicroscopy.org/` | Old challenge site retired |
| `https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/` | `https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection` | Page moved |
| `https://www.apple.com/healthcare/apple-heart-study/` | `https://med.stanford.edu/appleheartstudy.html` | Apple page removed; Stanford is the study host |
| `https://www.cms.gov/data-research/statistics-trends-and-reports` | `https://data.cms.gov/` | Old CMS data page removed |
| `https://www.cms.gov/medicare/quality` | `https://www.cms.gov/medicare/quality/measures` | Canonical quality-measures page |
| `https://www.ema.europa.eu/.../eudravigilance` | `https://www.ema.europa.eu/en/human-regulatory-overview/research-development/pharmacovigilance-research-development/eudravigilance` | EMA restructured the path |
| `https://aif360.res.ibm.com/` | `https://github.com/Trusted-AI/AIF360` | IBM demo site had TLS failure; GitHub is canonical |
| `https://gco.iarc.fr/` | `https://gco.iarc.who.int/` | IARC moved GCO to the WHO domain |
| `10.1016/j.orhc.2011.09.001` | `10.1007/s10729-010-9143-6` | Wrong DOI; corrected title/journal to *Health Care Management Science* |
| FDA AI/ML device page | `https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices` | Page moved |
| FDA QMSR | `https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr` | Page moved |
| FDA GMLP | `https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles` | 301 redirect |
| FDA FAERS | `https://www.fda.gov/drugs/surveillance-post-drug-approval-activities/fda-adverse-event-monitoring-system-aems` | FAERS renamed/merged into AEMS |
| CMS prior authorization | `https://www.cms.gov/priorities/electronic-prior-authorization/overview` | More specific current page |
| WESAD | `https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection` | Original page moved |
| PANDA | `https://panda.grand-challenge.org/data/` | Kaggle competition page retired or login-blocked |
| DECIDE-AI | `https://doi.org/10.1136/bmj-2022-070904` | Site unavailable; stable DOI |
| MERIT-HF | `10.1016/S0140-6736(99)04440-2` | Wrong DOI; Crossref-verified |
| All of Us v8 | CDR v9 figures | Dataset release updated June 2026 |
| GRAIL 2025 | GRAIL 2026 | Results reported May/September 2026 |

## 3. DOI verification

All 63 DOI citations were checked against Crossref with an extractor that preserves balanced parentheses. After correction of the operating-room review and MERIT-HF DOIs, **63/63 resolved to metadata consistent with the cited title and journal**. See `_research/doi_check.tsv` for the full results.

Notes:

- DOI resolution does not guarantee that the claim in the course matches the paper. Content-level checks were performed for key claims.
- Some publisher pages return 403 to automated clients even when the DOI is valid. That is a paywall/bot policy, not a bad citation.
- The two 2025 JAMA articles are cited by stable journal URLs rather than inferred DOIs.

## 4. Sources that require manual/browser verification

These are official or likely valid but blocked or slow from the automated client:

- FDA guidance pages: `www.fda.gov/...` (anti-bot response).
- FTC health privacy pages: `www.ftc.gov/...`.
- HCUP-US: `hcup-us.ahrq.gov` (timeout).
- IMDRF: `www.imdrf.org` (timeout).
- ML4H: `ml4h.cc` (TLS error).
- Sleep Data: `sleepdata.org` (TLS error).
- CIL Health: `www.cilhealth.org` (connection error).
- i2b2: `www.i2b2.org` (nonstandard 781 status; redirects to `i2b2.org`).
- PhysioNet and HIRID: occasional timeouts; retry or use the mirror if available.
- Box-hosted NIH ChestX-ray14: `nihcc.app.box.com/v/ChestXray-NIHCC` (Box blocks automated access; NIH and Google Cloud documentation confirm it is the canonical dataset).

## 5. Maintenance rule

Re-run the link checker quarterly. Treat 403/404 from FDA, FTC, and publisher sites as **needs manual check**, not automatically dead. For regulatory sources, prefer the stable guidance landing page plus the exact PDF title/date, because FDA restructures URLs without notice.
