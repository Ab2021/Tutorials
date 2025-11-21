# Day 48: Mock Interview 4 - Ad Click Prediction (Google/Ads)

> **Phase**: 5 - Interview Mastery
> **Week**: 10 - Mock Interviews
> **Focus**: High Scale & Calibration
> **Reading Time**: 60 mins

---

## 1. Problem Statement

"Predict the probability that a user clicks on an Ad. Used for pricing ($Cost = Bid \times pCTR$)."

---

## 2. Step-by-Step Design

### Step 1: Requirements
*   **Scale**: Billions of impressions per day.
*   **Calibration**: The probability *must* be accurate. If we predict 0.1, it should click 10% of the time.

### Step 2: Model
*   **Wide & Deep**:
    *   *Wide*: Linear model with Cross-Features (`User_City x Ad_Category`). Memorizes specific rules.
    *   *Deep*: DNN with Embeddings. Generalizes to unseen combinations.
*   **Why not just Deep?**: Deep models over-generalize. Wide part ensures we don't show "Wool Coats" in "Hawaii" just because "Coats" are popular globally.

### Step 3: Data
*   **Label**: Click (1) or No-Click (0).
*   **Delay**: Clicks happen seconds after view. Attribution window.

---

## 3. Deep Dive Questions

**Interviewer**: "The vocabulary of `User_ID` is 1 Billion. Embedding table won't fit in RAM."
**Candidate**: "Feature Hashing. Hash the ID to a space of $2^{20}$ (1 Million). Collisions will happen, but the Deep network can learn to disambiguate using other features. Or use a Bloom Filter to only learn embeddings for frequent users."

**Interviewer**: "Why is Calibration important?"
**Candidate**: "Because we charge advertisers based on expected clicks. If we predict 0.5 but reality is 0.1, we overcharge the advertiser (or overbid in the auction) and lose money/trust. We use Isotonic Regression to calibrate the raw model outputs."

---

## 4. Evaluation
*   **Metric**: LogLoss (Entropy). AUC is not enough (AUC cares about order, not absolute value).
*   **Calibration Plot**: Reliability Diagram.

---

## 5. Further Reading
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [Ad Click Prediction at Google](https://research.google/pubs/pub41159/)
