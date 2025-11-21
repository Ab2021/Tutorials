# Day 11 Interview Questions: AlexNet & VGG

## Q1: Why did AlexNet use a stride of 4 in the first layer?
**Answer:**
To aggressively downsample the high-resolution input ($227 \times 227$) early on.
*   Reduces computational cost/memory.
*   Increases receptive field quickly.
*   **Trade-off:** Loss of fine-grained spatial information.

## Q2: Explain the "VGG Block" design pattern.
**Answer:**
A sequence of Convolutional layers (usually $3 \times 3$, padding 1) followed by a Max Pooling layer ($2 \times 2$, stride 2).
*   Convolutions preserve spatial resolution ($H \times W$).
*   Pooling halves resolution ($H/2 \times W/2$).
*   Number of filters usually doubles after pooling to preserve information capacity.

## Q3: Why are $3 \times 3$ filters preferred over larger ones?
**Answer:**
1.  **Efficiency:** Fewer parameters for the same receptive field (stacking).
2.  **Non-linearity:** More layers = more activation functions = more expressive power.
3.  **Regularization:** Implicit regularization through decomposition.

## Q4: What is the receptive field of three stacked $3 \times 3$ convolutions?
**Answer:**
$7 \times 7$.
*   Layer 1: Sees $3 \times 3$.
*   Layer 2: Sees $3 \times 3$ of Layer 1 $\to 5 \times 5$.
*   Layer 3: Sees $3 \times 3$ of Layer 2 $\to 7 \times 7$.

## Q5: Why does VGG have so many parameters compared to ResNet?
**Answer:**
Because of the **Fully Connected (Dense) layers** at the end.
*   VGG flattens a $512 \times 7 \times 7$ feature map into a 25,088 vector and connects it to 4096 neurons. This single matrix multiplication accounts for ~100M parameters.
*   ResNet uses **Global Average Pooling** (GAP) to reduce $2048 \times 7 \times 7 \to 2048 \times 1 \times 1$, eliminating the massive dense layers.

## Q6: What was the main contribution of AlexNet to the field?
**Answer:**
It proved that Deep Convolutional Neural Networks could outperform classical CV methods (SIFT/HOG + SVM) by a significant margin on large-scale data (ImageNet), effectively starting the Deep Learning era.

## Q7: How do you calculate the output size of a Conv layer?
**Answer:**
$$ O = \frac{I - K + 2P}{S} + 1 $$
*   $I$: Input size
*   $K$: Kernel size
*   $P$: Padding
*   $S$: Stride

## Q8: Calculate parameters in a Conv layer with input $C_{in}=64$, output $C_{out}=128$, kernel $3 \times 3$.
**Answer:**
$$ \text{Params} = (\text{Kernel} \times C_{in} + \text{Bias}) \times C_{out} $$
$$ \text{Params} = (3 \times 3 \times 64 + 1) \times 128 $$
$$ = (576 + 1) \times 128 = 73,856 $$

## Q9: What is Overfitting? How did AlexNet prevent it?
**Answer:**
Overfitting is when the model learns the training data noise rather than the signal, performing poorly on test data.
**AlexNet solutions:**
1.  **Data Augmentation:** Random crops, flips, PCA color jitter.
2.  **Dropout:** Randomly disabling neurons (0.5 prob) in FC layers.

## Q10: Implement a VGG Block in PyTorch.
**Answer:**
```python
def vgg_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels # Next conv takes this output
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```
