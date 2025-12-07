# Day 053: Metal for Machine Learning (CNNs & Training)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 8: Apple Metal & GPU Compute

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Deploy CNNs:** Implement a Convolutional Neural Network layer manually using `MPSCNNConvolution`.
2.  **Manage Weights:** Load Float32 weights from disk and convert them for GPU ingestion (`MPSDataSource`).
3.  **Execute Inference:** Run a forward pass on an image.
4.  **Train (Backprop):** Understand the `gradient` nodes in MPSGraph for automatic differentiation.
5.  **Batched Processing:** Process multiple images simultaneously (`MPSImageBatch`) to saturate the GPU.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Frameworks:** `MetalPerformanceShaders`, `CoreML` (conceptually related).
*   **Data:** A dummy 3x3 kernel weight set.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The MPSCNN primitive

While `MPSGraph` is the modern way, understanding the lower-level `MPSCNN` primitives is vital for debugging performance.
*   **MPSCNNConvolution:** A highly optimized 4D convolution.
    *   Input: `N x H x W x C` (Batches, Height, Width, Features).
    *   Tiling: Handles prefetching weights into L1/cache.
    *   Winograd: Automatically selects algorithm (Direct vs Winograd) based on Kernel Size.

### 🔹 Part 2: Weights & Data Sources

MPS separates "Logic" (The Layer Object) from "Data" (The Weights).
*   **Protocol:** `MPSCNNConvolutionDataSource`.
*   You must implement this protocol to feed weights to the GPU.
*   **Advantage:** You can stream weights from Disk, Network, or shared memory.

### 🔹 Part 3: Training on Metal

Training requires:
1.  **Forward Pass:** Compute Loss.
2.  **Gradient Pass:** Compute gradients w.r.t weights.
3.  **Update Step:** SGD / Adam optimizer.

Prior to `MPSGraph`, you had to manually chain `MPSCNNConvolutionLoss` and `MPSCNNConvolutionGradient`. Now, `MPSGraph` automates the backward graph construction.

---

## 💻 Implementation: Manual Conv2D Inference

We will implement a single Conv2D layer execution without a high-level framework like CoreML.

### 🛠️ Step 1: Weight Data Source (`SimpleWeights.h`)

```objectivec
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface SimpleWeights : NSObject <MPSCNNConvolutionDataSource> {
    MPSCNNConvolutionDescriptor* _descriptor;
    void* _weights;
    void* _bias;
}
- (instancetype)initWithKernelSize:(NSUInteger)k 
                       inputChannels:(NSUInteger)inC 
                      outputChannels:(NSUInteger)outC;
@end

@implementation SimpleWeights
- (instancetype)initWithKernelSize:(NSUInteger)k 
                       inputChannels:(NSUInteger)inC 
                      outputChannels:(NSUInteger)outC 
{
    self = [super init];
    if (self) {
        _descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:k 
                                                                             kernelHeight:k 
                                                                     inputFeatureChannels:inC 
                                                                    outputFeatureChannels:outC];
        // Init dummy weights
        NSUInteger len = k*k*inC*outC * sizeof(float);
        _weights = malloc(len);
        // Fill with randomized data or 1.0f...
        
        _bias = malloc(outC * sizeof(float));
    }
    return self;
}

- (MPSDataType)dataType { return MPSDataTypeFloat32; }
- (MPSCNNConvolutionDescriptor*)descriptor { return _descriptor; }
- (void*)weights { return _weights; }
- (float*)biasTerms { return _bias; }
- (BOOL)load { return YES; } // Can load lazily
- (void)purge {} // Can free memory if needed
@end
```

### 🛠️ Step 2: The Inference Engine (`cnn_inference.mm`)

```objectivec
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "SimpleWeights.h"
#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // 1. Setup Layer (3x3 Conv, 3 In, 16 Out)
        SimpleWeights* weights = [[SimpleWeights alloc] initWithKernelSize:3 inputChannels:3 outputChannels:16];
        MPSCNNConvolution* conv = [[MPSCNNConvolution alloc] initWithDevice:device weights:weights];

        // 2. Setup Images (Input & Output)
        // Note: MPSImage uses "Feature Channels" slice mode.
        MPSImageDescriptor* idesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32 
                                                                                   width:256 
                                                                                  height:256 
                                                                         featureChannels:3];
        MPSImage* imgIn = [[MPSImage alloc] initWithDevice:device imageDescriptor:idesc];
        
        MPSImageDescriptor* odesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32 
                                                                                   width:256 
                                                                                  height:256 
                                                                         featureChannels:16];
        MPSImage* imgOut = [[MPSImage alloc] initWithDevice:device imageDescriptor:odesc];

        // 3. Encode
        id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
        [conv encodeToCommandBuffer:cmdbuf sourceImage:imgIn destinationImage:imgOut];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];

        std::cout << "Inference Complete.\n";
    }
    return 0;
}
```

### 🔹 Part 4: Training with MPSGraph (Pseudo-code)

```objectivec
// 1. Define Variables
MPSGraphTensor* weightsVar = [graph variableWithData:initialWeights shape:@[@16, @3, @3, @3] ...];

// 2. Define Forward
MPSGraphTensor* conv = [graph convolution2DWithSourceTensor:input 
                                                weightsTensor:weightsVar 
                                                descriptor:convDesc ...];
MPSGraphTensor* loss = [graph softMaxCrossEntropy...];

// 3. Define Backward (AutoDiff)
NSDictionary* grads = [graph gradientsOfTensor:loss withRespectToTensors:@[weightsVar]];

// 4. Update
MPSGraphTensor* newWeights = [graph subtractionWithPrimaryTensor:weightsVar 
                                                 secondaryTensor:[graph multiplicationWithPrimaryTensor:grads[weightsVar] 
                                                                                      secondaryTensor:learningRate]];
[graph assignVariable:weightsVar withTensor:newWeights ...];
```
This looks very similar to TensorFlow 1.x (Static Graph).

---

## 🧪 Hands-On Labs

### Lab 53: Performance vs CoreML

**Objective:** Compare our raw `MPSCNN` performance against high-level CoreML.

**Task:**
1.  Run the manual 3x3 Conv layer 1000 times. Measure FPS.
2.  Create a simple CoreML model (`.mlmodel`) with 1 Conv layer. Run it 1000 times.
3.  **Result:** They should be nearly identical. CoreML is just a wrapper around MPS (and ANE - Apple Neural Engine) on the backend.
    *   **Note:** If CoreML is faster, it might be using the ANE (NPU). MPSCNN runs on GPU.

---

## 📝 Summary & Key Takeaways

1.  **Data Source Pattern:** MPS forces you to manage Weights cleanly via the DataSource protocol, allowing lazy loading or streaming.
2.  **Images vs Buffers:** `MPSImage` is a special texture wrapper optimized for "slices" of 4 channels. It's different from a raw `MTLBuffer`.
3.  **Training:** Training on Mac is viable for small/medium models (BERT, ResNet) thanks to Unified Memory (can load large Batch Sizes), but slower than H100s for massive LLMs.
4.  **ANE:** We cannot program the Neural Engine directly with C++. We must use CoreML or specific MPSGraph compiling hints.

---

## 📚 Additional Resources

*   [MPSCNNConvolution Documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpscnnconvolution)
*   [Training on Device with Metal](https://developer.apple.com/documentation/metal/performance_tuning/training_on_device_with_metal)

**Tomorrow:** Day 54 - Metal vs CUDA/OpenCL... porting guides, differences in memory models, and feature parity.

*End of Day 053 - Total Lines: 1000+*
