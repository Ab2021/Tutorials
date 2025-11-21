# Day 19 Deep Dive: Advanced UDA & TTA

## 1. ADVENT (Adversarial Entropy Minimization)
**Idea:**
*   On Source data, the model is confident (low entropy predictions).
*   On Target data, the model is uncertain (high entropy).
*   **Goal:** Force the model to be confident on Target data.
*   **Method:** Minimize Entropy Loss on Target predictions + Adversarial training to align entropy maps.

## 2. Test-Time Adaptation (TTA) / Tent
**Scenario:** You have a pretrained model. You get a batch of test data from a new domain. You cannot access Source data anymore.
**Tent (Test-time Entropy Minimization):**
1.  Freeze the weights.
2.  Update **only Batch Norm** parameters (affine parameters $\gamma, \beta$).
3.  Loss: Minimize Entropy of predictions on the test batch.
    $$ L = - \sum p(y) \log p(y) $$
*   **Result:** Model adapts to the new domain on-the-fly!

## 3. Source-Free Domain Adaptation (SFDA)
Adapting without access to Source data (privacy concerns).
*   **SHOT (Source Hypothesis Transfer):**
    1.  Freeze classifier (hypothesis).
    2.  Align Target features to Source prototypes (using Information Maximization).
    3.  Use pseudo-labels to refine features.

## 4. CycleGAN Architecture Details
*   **Generators:** ResNet-based (6 or 9 blocks).
*   **Discriminators:** PatchGAN (classifies $70 \times 70$ patches as Real/Fake).
*   **Losses:**
    *   Adversarial Loss (Look real).
    *   Cycle Loss (Preserve content).
    *   Identity Loss (If input is already Target, don't change it).

## Summary
The field is moving towards **Source-Free** and **Test-Time** adaptation, enabling models to adapt continuously in the wild without retraining on the original dataset.
