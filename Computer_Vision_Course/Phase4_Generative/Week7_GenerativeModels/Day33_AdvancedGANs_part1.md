# Day 33 Deep Dive: StyleGAN Architecture & Inversion

## 1. The Mapping Network ($Z \to W$)
Why do we need it?
*   **Entanglement:** In $Z$ (Gaussian), attributes are entangled. "Men" might always have "Short Hair" due to dataset bias. Interpolating $z$ follows a curve on the sphere.
*   **Disentanglement:** The mapping network unwarps this space into $W$, where factors of variation are linear.
*   **Truncation Trick:** At test time, sample $w$ close to the average $\bar{w}$ to avoid weird artifacts (trade-off: less diversity).

## 2. Adaptive Instance Normalization (AdaIN)
*   Standard BatchNorm learns fixed $\mu, \sigma$ for the whole dataset.
*   AdaIN computes $\mu, \sigma$ dynamically from the style vector $w$.
*   It effectively "washes" the feature map to zero mean/unit var, then "paints" it with the statistics of the style.

## 3. GAN Inversion
**Goal:** Given a real image $x$, find the latent code $w$ such that $G(w) \approx x$.
*   **Optimization:** Freeze $G$, optimize $w$ to minimize $||G(w) - x||$.
*   **Encoder:** Train an encoder $E$ to predict $w$ from $x$.
*   **Application:** Image Editing.
    1.  Invert image to $w$.
    2.  Move $w$ in the "Smile" direction ($w' = w + \alpha n_{smile}$).
    3.  Generate $x' = G(w')$.

## 4. CycleGAN Identity Loss
*   What if we feed a Zebra image to the Horse $\to$ Zebra generator?
*   It should change nothing.
*   **Identity Loss:** $||G(y) - y||_1$.
*   Preserves color composition (prevents tinting).

## Summary
StyleGAN is not just a generator; it's a tool for understanding the manifold of images. GAN Inversion allows us to edit real images using the semantic knowledge learned by the GAN.
