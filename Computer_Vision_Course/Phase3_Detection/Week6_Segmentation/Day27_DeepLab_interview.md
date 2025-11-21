# Day 27 Interview Questions: DeepLab

## Q1: What is the main advantage of Atrous (Dilated) Convolution?
**Answer:**
It increases the **receptive field** of the filter without increasing the number of parameters or reducing the spatial resolution (no downsampling).
*   Allows the network to see a larger context while maintaining a dense feature map.

## Q2: Explain the "Gridding Artifact" in Dilated Convolutions.
**Answer:**
*   If we stack dilated convolutions with the same rate (e.g., $r=2, r=2, r=2$), the filter samples pixels in a checkerboard pattern and misses neighbors completely.
*   **Solution:** Use a "Hybrid Dilated Convolution" (HDC) strategy with varying rates (e.g., $1, 2, 3$) to cover all pixels.

## Q3: What is ASPP and why is it needed?
**Answer:**
**Atrous Spatial Pyramid Pooling.**
*   Objects in images appear at different scales (e.g., a person close to camera vs far away).
*   ASPP applies filters with different dilation rates ($r=6, 12, 18$) in parallel.
*   This effectively captures features at multiple scales (small field of view vs large field of view) and fuses them.

## Q4: Why did DeepLab v3+ add a Decoder?
**Answer:**
*   DeepLab v3 output was at stride 16 (coarse).
*   Upsampling by 16x (Bilinear) loses boundary details.
*   The Decoder (similar to U-Net) fuses the semantic features from the encoder with low-level features (stride 4) to refine the boundaries and produce a sharper mask (stride 4).

## Q5: What is the difference between Global Average Pooling and Pyramid Pooling?
**Answer:**
*   **Global Avg Pooling:** Reduces the entire feature map to a single $1 \times 1$ vector. Captures "what" is in the image but loses "where".
*   **Pyramid Pooling (PSPNet):** Pools at multiple grid sizes ($1 \times 1, 2 \times 2, \dots$). Captures context at different levels of granularity.

## Q6: How does CRF improve segmentation?
**Answer:**
*   CNN predictions are often "blobby" and don't respect object edges perfectly.
*   CRF (Conditional Random Field) enforces a smoothness constraint: "Pixels that are close and have similar color should have the same label."
*   This forces the segmentation mask to snap to the strong edges in the image.

## Q7: Calculate the receptive field of a $3 \times 3$ dilated conv with rate $r=2$.
**Answer:**
*   The kernel covers a region of size $k_{eff} = k + (k-1)(r-1)$.
*   $3 + (2)(1) = 5$.
*   It acts like a $5 \times 5$ kernel but with only 9 weights (sparse).

## Q8: Why is "Output Stride" important in segmentation?
**Answer:**
*   It determines the density of the prediction.
*   Stride 32 (ResNet) means 1 prediction for every $32 \times 32$ pixel block. Too coarse for segmentation.
*   Stride 8 (DeepLab) means 1 prediction for every $8 \times 8$ block. Much finer.
*   Reducing stride requires removing pooling layers and using dilated convolutions to maintain receptive field.

## Q9: What is "Poly Learning Rate Policy"?
**Answer:**
A learning rate schedule often used in segmentation (DeepLab/PSPNet).
$$ lr = lr_{base} \times (1 - \frac{iter}{max\_iter})^{power} $$
*   Usually $power=0.9$.
*   Decays LR smoother than step decay and reaches 0 at the end.

## Q10: Implement a Dilated Conv block in PyTorch.
**Answer:**
```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
```
*   Note: `padding` must equal `dilation` to maintain spatial size (if stride=1).
