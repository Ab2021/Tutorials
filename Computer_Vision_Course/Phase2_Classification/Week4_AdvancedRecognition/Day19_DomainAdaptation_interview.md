# Day 19 Interview Questions: Domain Adaptation

## Q1: What is the "Domain Gap"?
**Answer:**
The difference in data distribution between the training set (Source) and the test set (Target).
*   Example: Training on clear weather images, testing on rainy weather images.
*   This gap causes a performance drop because the model overfits to Source-specific statistics.

## Q2: Explain the Gradient Reversal Layer (GRL).
**Answer:**
A layer used in DANN.
*   **Forward:** Acts as Identity ($x \to x$).
*   **Backward:** Flips the sign of the gradient ($g \to -\lambda g$).
*   It is placed between the Feature Extractor and the Domain Classifier.
*   It forces the Feature Extractor to learn features that maximize the Domain Classifier's loss (i.e., features that make it impossible to tell if the image is Source or Target).

## Q3: Why do we need Cycle Consistency Loss in CycleGAN?
**Answer:**
*   Standard GAN loss only ensures the output looks like the Target domain (e.g., "make it look like a zebra").
*   It doesn't ensure the content is preserved (e.g., the zebra might be in a different position than the original horse).
*   **Cycle Loss** ($F(G(x)) \approx x$) forces the mapping to be reversible, ensuring that the geometric structure and content are preserved during translation.

## Q4: What is "Entropy Minimization" in Domain Adaptation?
**Answer:**
*   Assumption: A good model should be confident (low entropy) in its predictions.
*   On Target data, models are often uncertain (high entropy).
*   By explicitly minimizing the entropy of predictions on unlabeled Target data, we force the decision boundaries to move away from dense regions of data, effectively aligning the classes.

## Q5: What is Test-Time Adaptation (TTA)?
**Answer:**
Adapting a model to a new domain **during inference**, using only the incoming test data.
*   Typically involves updating a small set of parameters (like Batch Norm stats) to minimize an unsupervised loss (like Entropy).
*   Does not require Source data or labels.

## Q6: Difference between Feature Alignment and Pixel Alignment?
**Answer:**
*   **Feature Alignment (DANN):** Align distributions in the high-level feature space (at the end of the CNN).
*   **Pixel Alignment (CycleGAN):** Align distributions in the raw pixel space (transform the image itself).

## Q7: What is "Covariate Shift"?
**Answer:**
A type of domain shift where the input distribution $P(X)$ changes, but the conditional distribution of labels $P(Y|X)$ remains the same.
*   Example: The objects (X) look different (cartoon vs real), but a "dog" is still a "dog" (Y).

## Q8: Why is Batch Normalization important in Domain Adaptation?
**Answer:**
*   BN stores running mean/variance of the training data (Source).
*   When testing on Target data, these statistics might be wrong.
*   **Adaptive BN:** Re-calculating BN statistics on the Target data is a simple but very effective form of adaptation.

## Q9: What is "Pseudo-Labeling"?
**Answer:**
1.  Train model on Labeled Source.
2.  Predict labels for Unlabeled Target.
3.  Select high-confidence predictions as "Pseudo-Labels".
4.  Retrain model on Source + Pseudo-Labeled Target.
*   Iterative process.

## Q10: Implement Gradient Reversal in PyTorch.
**Answer:**
```python
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```
