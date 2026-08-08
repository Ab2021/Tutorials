# AIMET APIs and Configuration Reference

Welcome to the **AIMET APIs and Configuration Reference**. This comprehensive, deep-dive document is designed to serve as an exhaustive guide to the AI Model Efficiency Toolkit (AIMET) provided by Qualcomm. AIMET is a powerful library that provides advanced quantization and compression techniques for neural network models. 

This document covers everything from detailed API references for PyTorch, TensorFlow, and ONNX to JSON configuration file schemas, optimization techniques like Cross-Layer Equalization (CLE), Adaptive Rounding (AdaRound), and various model compression algorithms like Spatial SVD and Channel Pruning. 

---

## 1. Complete QuantizationSimModel API Reference

The `QuantizationSimModel` (QuantSim) API is the cornerstone of AIMET's quantization capabilities. It allows you to simulate quantization noise during model training or evaluation. By wrapping your model in a QuantSim object, AIMET inserts quantizer and dequantizer nodes into the computational graph.

### PyTorch API

```python
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme

quant_sim = QuantizationSimModel(
    model=model,                  # The PyTorch model to quantize
    dummy_input=dummy_input,      # Dummy input tensor for graph tracing
    quant_scheme=QuantScheme.post_training_tf_enhanced, # Quantization scheme
    rounding_mode='nearest',      # Rounding mode
    default_output_bw=8,          # Default bitwidth for activation tensors
    default_param_bw=8,           # Default bitwidth for parameter tensors
    in_place=False,               # If True, modifies the model in place
    config_file='quantsim_config.json' # Path to a JSON configuration file
)
```

### Parameters Breakdown:
- **`model`** (`torch.nn.Module`): The base PyTorch model you wish to quantize.
- **`dummy_input`** (`Union[torch.Tensor, Tuple]`): A sample input or tuple of inputs matching the model's expected input shape. This is critical as AIMET uses PyTorch's JIT tracer to build a directed acyclic graph (DAG) of the model.
- **`quant_scheme`** (`aimet_common.defs.QuantScheme`): Determines how the quantization parameters (min/max or scale/offset) are computed. Options include:
  - `QuantScheme.post_training_tf`: Classic TensorFlow quantization scheme.
  - `QuantScheme.post_training_tf_enhanced`: Enhanced TF scheme that minimizes the SQNR (Signal-to-Quantization-Noise Ratio).
  - `QuantScheme.training_range_learning_with_tf_init`: For Quantization-Aware Training (QAT), where the min/max ranges are learned during backpropagation.
  - `QuantScheme.post_training_percentile`: Uses percentile-based calibration for setting the min/max ranges.
- **`rounding_mode`** (`str`): The rounding mode for the quantizers. Supported values include `'nearest'` and `'stochastic'`.
- **`default_output_bw`** (`int`): Bitwidth for all activation tensors. Defaults to 8.
- **`default_param_bw`** (`int`): Bitwidth for all weight and bias parameters. Defaults to 8.
- **`in_place`** (`bool`): If True, modifies the original `model` directly. If False, operates on a deep copy of the model.
- **`config_file`** (`str`): Path to a JSON file specifying granular quantization rules (e.g., leaving certain layers in FP16/FP32, enforcing asymmetric quantization).

---

## 2. `compute_encodings()` - Options and Callback Patterns

Once the `QuantSim` object is created, it has no knowledge of the optimal quantization ranges. You must compute these encodings by passing representative data through the model.

### API Signature
```python
quant_sim.compute_encodings(
    forward_pass_callback=forward_pass_callback_fn,
    forward_pass_callback_args=callback_args
)
```

### Callback Pattern
AIMET requires a callback function to run data through the network. The user defines this function to process a limited number of batches (typically 500-1000 images are sufficient for Post-Training Quantization calibration).

```python
def forward_pass_callback_fn(model, args):
    """
    User-defined callback to pass data through the model.
    model: The wrapped QuantSim model
    args: User arguments (e.g., DataLoader, number of batches, device)
    """
    data_loader = args['data_loader']
    num_batches = args['num_batches']
    device = args['device']
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            model(inputs)
            if i >= num_batches - 1:
                break

# Preparing arguments and invoking compute_encodings
callback_args = {
    'data_loader': calibration_data_loader,
    'num_batches': 100,
    'device': torch.device('cuda')
}

quant_sim.compute_encodings(forward_pass_callback_fn, callback_args)
```
During this process, AIMET monitors the statistics (min, max, histogram) of the activations at each layer and computes the final scaling factors and offsets according to the selected `QuantScheme`.

---

## 3. `export()` - Options, ONNX vs PyTorch, Encoding Format

After the model is quantized and fine-tuned, the final step is to export the model and its quantization encodings to a format that can be ingested by hardware accelerators (e.g., Qualcomm Hexagon DSP / SNPE).

### API Signature
```python
quant_sim.export(
    path='./exported_model',
    filename_prefix='resnet18_quantized',
    dummy_input=dummy_input,
    onnx_export_args=onnx_export_args
)
```

### Export Artifacts
Calling `export()` generates two main artifacts:
1. **The Model**: Either an ONNX file (`resnet18_quantized.onnx`) or a PyTorch TorchScript file (`resnet18_quantized.pt`), depending on the framework.
2. **The Encodings JSON**: A file named `resnet18_quantized.encodings` containing the scale, offset, min, max, and bitwidth for every tensor in the model.

### ONNX Export Arguments
When exporting PyTorch models to ONNX, you can pass arguments native to `torch.onnx.export` via `onnx_export_args`:

```python
onnx_export_args = {
    'opset_version': 11,
    'input_names': ['input_tensor'],
    'output_names': ['output_tensor'],
    'dynamic_axes': {
        'input_tensor': {0: 'batch_size'},
        'output_tensor': {0: 'batch_size'}
    }
}

quant_sim.export(
    path='./export',
    filename_prefix='my_model',
    dummy_input=dummy_input,
    onnx_export_args=onnx_export_args
)
```

### Encoding Format Interpretation
The generated `*.encodings` file is a JSON file structure. Hardware compilers read this file to assign fixed-point types.

```json
{
  "activation_encodings": {
    "Conv1_output_tensor": [
      {
        "bitwidth": 8,
        "max": 5.4,
        "min": 0.0,
        "offset": 0,
        "scale": 0.021176470588235293,
        "is_symmetric": "False"
      }
    ]
  },
  "param_encodings": {
    "Conv1.weight": [ ... ]
  }
}
```

---

## 4. `set_and_freeze_param_encodings()` - Use Cases

Sometimes, you need to manually intervene in the quantization process, specifically by freezing the quantization parameters of weights after they have been computed (e.g., after AdaRound).

```python
quant_sim.set_and_freeze_param_encodings(encoding_path='./adaround_encodings.json')
```

### Use Cases:
1. **Integrating AdaRound Results**: AdaRound computes highly optimized weight encodings. To evaluate the AdaRounded model in a QAT or standard QuantSim pipeline, you load the AdaRound encodings and freeze them so they aren't overwritten by `compute_encodings()`.
2. **Transfer Learning / Fine-Tuning**: If you are fine-tuning a pre-quantized backbone, you might freeze the quantization parameters of the early layers to preserve their stability while only updating the final classifier layer.
3. **Debugging / Isolation**: To isolate whether accuracy drop is caused by activation quantization or weight quantization, you can freeze weight encodings to perfect fp32 ranges and only test activation quantization.

---

## 5. AutoQuant API - Parameters and Return Values

`AutoQuant` is an end-to-end automated API introduced by AIMET to simplify the quantization workflow. Instead of manually applying Cross-Layer Equalization (CLE), AdaRound, and evaluating the model, `AutoQuant` automatically searches for the best combination of these techniques to achieve target accuracy.

```python
from aimet_torch.auto_quant import AutoQuant

auto_quant = AutoQuant(
    allowed_accuracy_drop=0.01, # 1% allowable drop
    unlabeled_dataset_iterable=data_loader,
    eval_callback=eval_callback,
    default_output_bw=8,
    default_param_bw=8
)

# Set optional parameters
auto_quant.set_adaround_params(adaround_params)

# Execute the AutoQuant pipeline
model, accuracy, encoding_path = auto_quant.apply(
    model, 
    dummy_input_batched=dummy_input
)
```

### Parameters:
- **`allowed_accuracy_drop`** (`float`): The maximum acceptable drop in accuracy. `AutoQuant` will stop early if a fast technique (like CLE) meets this target, saving time on expensive techniques like AdaRound.
- **`unlabeled_dataset_iterable`** (`Iterable`): Used for calibration in CLE and AdaRound.
- **`eval_callback`** (`Callable`): A function that takes a PyTorch model and returns a scalar metric (e.g., top-1 accuracy).

### Return Values:
1. **`model`**: The best quantized model found during the pipeline.
2. **`accuracy`**: The evaluation metric score of the returned model.
3. **`encoding_path`**: Path to the final encodings JSON file for the model.

---

## 6. AdaroundParameters - All Options Explained

Adaptive Rounding (AdaRound) formulates weight rounding as a localized optimization problem. Instead of blindly using round-to-nearest, AdaRound learns whether it is better to round a weight up or down to minimize the output difference at a layer level.

```python
from aimet_torch.adaround.adaround_weight import AdaroundParameters

adaround_params = AdaroundParameters(
    data_loader=calibration_data_loader,
    num_batches=20,
    default_num_iterations=10000,
    default_reg_param=0.01,
    default_beta_range=(20, 2),
    default_warm_start=0.2
)
```

### Parameters Breakdown:
- **`data_loader`**: Provides the calibration data to compute the layer-wise activations.
- **`num_batches`**: Number of batches from the data_loader to use. Usually, ~1000 samples (e.g., 32 batches of size 32) are sufficient.
- **`default_num_iterations`**: The number of optimization iterations for the AdaRound loss function per layer. 10000 is the standard default.
- **`default_reg_param`**: Regularization parameter that controls the penalty applied to the rounding variable.
- **`default_beta_range`**: Tuple defining the start and end beta temperatures for the annealing schedule. This controls how the continuous variables snap to discrete 0/1 bounds.
- **`default_warm_start`**: Fraction of total iterations dedicated to a warm start (where no regularization is applied), allowing the weights to drift towards optimal continuous values before snapping.

---

## 7. BN Folding APIs

Batch Normalization (BN) folding is a crucial pre-processing step. It absorbs the scale and shift parameters of BN layers into the preceding Convolutional or Linear layers. This improves inference speed and is mandatory before applying Cross-Layer Equalization (CLE).

### PyTorch API
```python
from aimet_torch.batch_norm_fold import fold_all_batch_norms

# In-place operation to fold all eligible BN layers
folded_pairs = fold_all_batch_norms(
    model, 
    input_shapes=(1, 3, 224, 224)
)

print(f"Folded {len(folded_pairs)} BN layers.")
```

### prepare_model API (PyTorch specific)
PyTorch models often contain operations that are not defined as `nn.Module` (e.g., `torch.add`, `F.relu`). `prepare_model` modifies the graph to replace these functional calls with AIMET-compatible Module equivalents, essential for successful BN folding and QuantSim tracing.

```python
from aimet_torch.model_preparer import prepare_model

# Prepare the model for AIMET
prepared_model = prepare_model(model)
fold_all_batch_norms(prepared_model, input_shapes=(1, 3, 224, 224))
```

---

## 8. CLE APIs - Equalize Model and Cross-Layer Scaling

Cross-Layer Equalization (CLE) tackles the issue of varying dynamic ranges across channels in depthwise separable convolutions, specifically prevalent in MobileNet architectures. It scales weights mathematically between adjacent layers so that the quantization ranges are more uniform.

### equalize_model API

```python
from aimet_torch.cross_layer_equalization import equalize_model

# Assumes fold_all_batch_norms has already been called
equalize_model(
    model, 
    input_shapes=(1, 3, 224, 224)
)
```

### get_cross_layer_scaling_params

If you wish to inspect or manually apply the scale factors, you can calculate them directly.

```python
from aimet_torch.cross_layer_equalization import get_cross_layer_scaling_params

# Find the scale factors between two connected layers
# e.g., Conv1 -> Relu -> Conv2
scale_factors = get_cross_layer_scaling_params(
    layer1=conv1_module, 
    layer2=conv2_module
)
```

---

## 9. Bias Correction APIs

Quantization introduces systematic bias (a shift in the mean of the activations). Bias Correction empirically estimates this shift and subtracts it by updating the bias parameter of the convolutional or linear layers.

```python
from aimet_torch.bias_correction import correct_bias

# Correct bias requires the quantized model and the original model
correct_bias(
    model.eval(),
    quant_params, # from quant_sim
    num_quant_samples=1000,
    data_loader=calibration_data_loader,
    num_batches=32
)
```
*Note: Bias Correction is highly recommended for models containing Depthwise Convolutions without applying CLE, or as a post-CLE cleanup.*

---

## 10. Compression APIs - ModelCompressor

AIMET supports multiple structural compression techniques designed to reduce MACs (Multiply-Accumulate operations) and parameter counts.

### Spatial SVD
Spatial Singular Value Decomposition splits a large convolution kernel into two smaller, cascaded kernels.

```python
from aimet_torch.compress import ModelCompressor
from aimet_torch.defs import SpatialSvdParameters

svd_params = SpatialSvdParameters(
    target_comp_ratio=0.5,           # Compress model to 50% of original MACs
    num_comp_ratio_candidates=3,     # Candidates for greedy search
    modules_to_ignore=[model.conv1], # Skip early layers
    condition='manual',
    manual_params={'conv2': 0.5}     # Per-layer specific ratios
)

compressed_model, stats = ModelCompressor.compress_model(
    model=model,
    eval_callback=eval_callback,
    eval_iterations=5,
    input_shape=(1, 3, 224, 224),
    compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
    cost_metric=aimet_common.defs.CostMetric.mac,
    parameters=svd_params
)
```

### Channel Pruning
Channel Pruning eliminates entirely unnecessary channels (filters) based on their impact on reconstruction error.

```python
from aimet_torch.defs import ChannelPruningParameters

pruning_params = ChannelPruningParameters(
    data_loader=calibration_data_loader,
    num_reconstruction_samples=500,
    allow_custom_downsample_ops=True,
    target_comp_ratio=0.6,
    num_comp_ratio_candidates=3,
    modules_to_ignore=[model.fc]
)

compressed_model, stats = ModelCompressor.compress_model(
    model=model,
    eval_callback=eval_callback,
    eval_iterations=5,
    input_shape=(1, 3, 224, 224),
    compress_scheme=aimet_common.defs.CompressionScheme.channel_pruning,
    cost_metric=aimet_common.defs.CostMetric.mac,
    parameters=pruning_params
)
```

---

## 11. QuantAnalyzer API

`QuantAnalyzer` is a critical diagnostic tool. It automatically runs a suite of analyses to determine which layers are most sensitive to quantization and which quantization configuration (CLE, AdaRound, FP16 vs INT8) provides the best results.

```python
from aimet_torch.quant_analyzer import QuantAnalyzer

analyzer = QuantAnalyzer(
    model=model,
    dummy_input=dummy_input,
    forward_pass_callback=forward_pass_callback_fn,
    eval_callback=eval_callback
)

# Enable/Disable specific analysis modules
analyzer.enable_per_layer_mse_loss(unlabeled_dataset_iterable=data_loader, num_batches=10)
analyzer.enable_per_layer_min_max_ranges()

# Run the analyzer
analyzer.analyze(
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    default_param_bw=8,
    default_output_bw=8,
    config_file='default_config.json',
    results_dir='./quant_analyzer_results'
)
```
**Results:** The output includes HTML files with plots showing MSE loss per layer, weight range distributions, and recommendations on which layers to keep in FP16 to restore accuracy.

---

## 12. Full JSON Configuration File Reference

The QuantSim JSON configuration file provides granular control over the quantization process. It defines defaults, overrides for specific layers, and operator-level rules.

```json
{
  "defaults": {
    "ops": {
      "is_output_quantized": "True"
    },
    "params": {
      "is_quantized": "True",
      "is_symmetric": "True"
    },
    "per_channel_quantization": "False",
    "strict_symmetric": "False"
  },
  "params": {
    "bias": {
      "is_quantized": "False"
    }
  },
  "op_type": {
    "Conv": {
      "is_input_quantized": "True",
      "is_symmetric": "False",
      "params": {
        "weight": {
          "is_quantized": "True",
          "is_symmetric": "True"
        }
      }
    },
    "Gemm": {
      "is_output_quantized": "False"
    }
  },
  "supergroups": [
    {
      "op_list": ["Conv", "BatchNormalization", "Relu"]
    },
    {
      "op_list": ["Conv", "Relu"]
    },
    {
      "op_list": ["Gemm", "Relu"]
    }
  ],
  "model_input": {
    "is_input_quantized": "True"
  },
  "model_output": {
    "is_output_quantized": "True"
  }
}
```

### Breakdown of JSON Fields
- **`defaults`**: The base rules applied to all nodes unless overridden.
  - `strict_symmetric`: If True, for symmetric quantization, the negative and positive bounds are identical exactly (e.g., -127 to 127 for 8-bit).
- **`params`**: Rules specific to parameter types across the network (e.g., globally turn off bias quantization).
- **`op_type`**: Overrides targeted at specific operation types (e.g., Convolution vs Matrix Multiplication).
- **`model_input` / `model_output`**: Controls whether the absolute input/output tensors of the model should have quantizer nodes attached.

---

## 13. Supergroups Configuration

In the JSON configuration, `supergroups` define sequences of operations that hardware accelerators (like DSPs) fuse into a single block. 

### Why use Supergroups?
When a Convolution, Batch Normalization, and ReLU are executed on hardware, they are usually fused. The intermediate tensors (e.g., between Conv and ReLU) are held in high-precision registers and are *not* quantized to 8-bit memory. 
If AIMET simulates quantization noise on these intermediate tensors, it introduces fake noise that the hardware won't experience, resulting in a pessimistic accuracy drop.

```json
"supergroups": [
  {
    "op_list": ["Conv", "BatchNormalization", "Relu"]
  },
  {
    "op_list": ["Conv", "Relu"]
  }
]
```
**Effect**: With this configured, AIMET will *not* insert an activation quantizer at the output of the `Conv` or the `BatchNormalization`. It will only place an activation quantizer at the output of the final `Relu`.

---

## 14. Per-channel Quantization Configuration

Per-channel (or per-axis) quantization applies a unique scale and offset to each output channel of a Convolution layer's weights. This drastically reduces quantization error for layers with high variance across filters.

To enable this, update the JSON configuration file:

```json
{
  "defaults": {
    "per_channel_quantization": "True"
  },
  "op_type": {
    "Conv": {
      "per_channel_quantization": "True"
    }
  }
}
```

**Implementation Detail**: When exported, the `.encodings` file will show arrays of scales and offsets equal to the number of output channels, rather than a scalar value.

```json
"weight": [
  {
    "bitwidth": 8,
    "max": 1.5,
    "min": -1.5,
    "scale": 0.0117,
    "offset": 0
  },
  {
    "bitwidth": 8,
    "max": 3.0,
    "min": -3.0,
    "scale": 0.0235,
    "offset": 0
  }
  ...
]
```

---

## 15. Custom Quantizers and Overrides

AIMET allows for node-specific overrides via Python code. You can target specific modules in the model and change their quantization bitwidth or bypass them entirely.

```python
# Assuming quant_sim is an initialized QuantizationSimModel object

# Find a specific module (e.g., the final linear layer)
fc_layer = quant_sim.model.fc

# 1. Bypass quantization for this specific layer
fc_layer.output_quantizers[0].enabled = False
fc_layer.input_quantizers[0].enabled = False
fc_layer.param_quantizers['weight'].enabled = False

# 2. Change bitwidth for a specific layer to 16-bit
fc_layer.output_quantizers[0].bitwidth = 16
fc_layer.param_quantizers['weight'].bitwidth = 16

# 3. Change quant scheme on the fly
fc_layer.param_quantizers['weight'].quant_scheme = QuantScheme.post_training_percentile
```

---

## 16. TensorFlow API Equivalents

AIMET provides parity across PyTorch and TensorFlow (specifically TF 2.x and Keras).

### TensorFlow QuantSim
```python
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme

quant_sim = QuantizationSimModel(
    model=keras_model,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    rounding_mode='nearest',
    default_output_bw=8,
    default_param_bw=8
)

# Compute encodings callback in TF is slightly different; it takes a tf.data.Dataset
quant_sim.compute_encodings(
    forward_pass_callback=lambda model, dataset: model.predict(dataset),
    forward_pass_callback_args=calibration_dataset
)

quant_sim.export(path='./tf_export', filename_prefix='tf_model')
```

### TensorFlow CLE
```python
from aimet_tensorflow.keras.cross_layer_equalization import equalize_model

cle_model = equalize_model(keras_model)
```

### TensorFlow AdaRound
```python
from aimet_tensorflow.keras.adaround_weight import Adaround, AdaroundParameters

adaround_model = Adaround.apply_adaround(
    keras_model, 
    params=adaround_params, 
    path='./tf_adaround', 
    filename_prefix='tf_adaround'
)
```

---

## 17. ONNX API Reference

AIMET also supports direct optimization of ONNX graphs. This is particularly useful for models exported from frameworks other than PyTorch or TensorFlow.

### ONNX QuantSim
```python
from aimet_onnx.quantsim import QuantizationSimModel
import onnx

onnx_model = onnx.load("model.onnx")

quant_sim = QuantizationSimModel(
    model=onnx_model,
    dummy_input=dummy_input_dict, # Dictionary mapping input names to np arrays
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    default_output_bw=8,
    default_param_bw=8,
    config_file='quantsim_config.json'
)

quant_sim.compute_encodings(forward_pass_callback, callback_args)
quant_sim.export(path='./onnx_out', filename_prefix='model_quantized')
```

### ONNX CLE
```python
from aimet_onnx.cross_layer_equalization import CrossLayerEqualization

# In ONNX, CLE modifies the graph and returns it
cle_model = CrossLayerEqualization.equalize_model(onnx_model)
onnx.save(cle_model, "model_cle.onnx")
```

---

## 18. Common Code Patterns and Idioms

### The "Golden Workflow" for INT8 Inference
The most successful and robust pipeline for deploying networks to 8-bit hardware using AIMET involves the following steps:

1. **Prepare Model & BN Folding**: Always run this first.
2. **Cross-Layer Equalization (CLE)**: Essential for models with depthwise convolutions (MobileNets, EfficientNets).
3. **AdaRound**: Apply AdaRound for weight quantization recovery.
4. **QuantSim Evaluation**: Wrap the AdaRounded model in QuantSim to simulate activation quantization and verify accuracy.

### Code Snippet of Golden Workflow
```python
import copy
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quantsim import QuantizationSimModel

# 1. Prepare and Fold
model = prepare_model(original_model)
model.eval()
fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))

# 2. CLE
equalize_model(model, input_shapes=(1, 3, 224, 224))

# 3. AdaRound
adaround_params = AdaroundParameters(
    data_loader=calib_loader, num_batches=32, default_num_iterations=10000
)
adaround_model = Adaround.apply_adaround(
    model, dummy_input, adaround_params, path='./adaround', filename_prefix='model'
)

# 4. QuantSim Evaluation
quant_sim = QuantizationSimModel(
    model=adaround_model,
    dummy_input=dummy_input,
    default_output_bw=8, default_param_bw=8
)

# Freeze the AdaRound parameter encodings so they aren't overwritten
quant_sim.set_and_freeze_param_encodings('./adaround/model.encodings')

# Compute activation encodings
quant_sim.compute_encodings(forward_pass_callback, callback_args)

# Final export
quant_sim.export('./final_export', 'quantized_model', dummy_input)
```

This workflow ensures minimal accuracy loss while strictly adhering to INT8 precision bounds.

---

## 19. Complete QuantSim TensorFlow API reference
```python
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme

quant_sim = QuantizationSimModel(
    model=keras_model,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    rounding_mode='nearest',
    default_output_bw=8,
    default_param_bw=8,
    config_file='quantsim_config.json'
)

def forward_pass_callback(model, dataset):
    model.predict(dataset)

quant_sim.compute_encodings(
    forward_pass_callback=forward_pass_callback,
    forward_pass_callback_args=my_tf_dataset
)

quant_sim.export(path='./tf_export', filename_prefix='tf_model')
```

---

## 20. Complete ONNX quantization API reference
```python
from aimet_onnx.quantsim import QuantizationSimModel
import onnx
import numpy as np
from aimet_common.defs import QuantScheme

onnx_model = onnx.load("model.onnx")
dummy_input = {'input': np.random.randn(1, 3, 224, 224).astype(np.float32)}

quant_sim = QuantizationSimModel(
    model=onnx_model,
    dummy_input=dummy_input,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    default_output_bw=8,
    default_param_bw=8
)

def forward_pass_callback(session, args):
    session.run(None, {'input': args})

quant_sim.compute_encodings(forward_pass_callback, dummy_input['input'])
quant_sim.export(path='./onnx_out', filename_prefix='onnx_quantized')
```

---

## 21. Advanced QuantSim: custom quantizers and overrides
You can programmatically access the quantizers within the QuantSim object to apply custom settings.
```python
for name, module in quant_sim.model.named_modules():
    if 'conv1' in name:
        module.output_quantizers[0].bitwidth = 16
        module.output_quantizers[0].is_symmetric = False
        module.param_quantizers['weight'].enabled = False
```

---

## 22. Mixed precision API: setting different bit-widths per layer
AIMET supports mixed precision quantization, where different layers operate at different bit-widths to balance accuracy and latency.
```python
from aimet_torch.mixed_precision import choose_mixed_precision

choose_mixed_precision(
    quant_sim=quant_sim,
    eval_callback=lambda *args, **kwargs: 0.95,
    eval_req=0.99, # Require 99% of original FP32 accuracy
    candidates=[(8, 8), (16, 8), (16, 16)], # (activation_bw, weight_bw)
    amp_search_algo='interpolation'
)
```

---

## 23. BN Re-estimation API with detailed code
BN Re-estimation recalculates the running mean and variance of Batch Normalization layers on the quantized model.
```python
from aimet_torch.bn_reestimation import reestimate_bn_stats

reestimate_bn_stats(
    model=quant_sim.model,
    dataloader=calibration_loader,
    num_batches=100,
    forward_fn=forward_pass_callback_fn
)
```

---

## 24. QuantAnalyzer detailed API (all methods, parameters, output format)
`QuantAnalyzer` runs a comprehensive suite of analyses.
- `enable_per_layer_mse_loss()`: Computes MSE between FP32 and INT8 activations.
- `enable_per_layer_min_max_ranges()`: Tracks the dynamic range of tensors.
- `enable_per_layer_snr()`: Calculates Signal-to-Noise Ratio.
Outputs are saved as interactive HTML plots in the specified `results_dir`.

---

## 25. AIMET logging and debugging APIs
AIMET uses standard Python logging. To enable debug logs:
```python
import logging
from aimet_common.utils import AimetLogger

AimetLogger.set_level_for_all_areas(logging.DEBUG)
```

---

## 26. Model comparison utilities
AIMET provides utilities to compare intermediate feature maps of the original and quantized models.
```python
from aimet_torch.visualize_model import visualize_relative_weight_ranges_to_identify_problematic_layers

visualize_relative_weight_ranges_to_identify_problematic_layers(
    model=quant_sim.model,
    results_dir='./debug_viz'
)
```

---

## 27. AIMET with DataParallel and DistributedDataParallel (multi-GPU)
When using QAT, the QuantSim model can be wrapped in `DataParallel` or `DistributedDataParallel` just like a standard PyTorch model.
```python
import torch
quant_sim.model = torch.nn.DataParallel(quant_sim.model)
# Proceed with standard QAT training loop
```

---

## 28. Integration with PyTorch Lightning
AIMET QuantSim models are standard `torch.nn.Module`s, making them fully compatible with PyTorch Lightning modules. You simply initialize the QuantSim object inside the `setup()` or `__init__()` of your LightningModule.

---

## 29. Integration with Hugging Face Trainer
Similarly, the `quant_sim.model` can be passed directly to the Hugging Face `Trainer` API for QAT on transformer models.

---

## 30. AIMET environment variables and runtime configuration
AIMET relies on several environment variables for backend configuration:
- `CUDA_VISIBLE_DEVICES`: Controls GPU visibility.
- `AIMET_LOG_LEVEL`: Sets the default logging level.

---

## 31. Performance profiling: AIMET-specific profiling hooks
You can attach PyTorch hooks to the QuantSim model to profile the latency overhead of the fake quantization nodes during QAT.

---

## 32. Serialization and loading of quantized models
To save and load a QAT checkpoint:
```python
# Save
torch.save(quant_sim.model.state_dict(), 'qat_ckpt.pth')

# Load
# quant_sim_new = QuantizationSimModel(...)
# quant_sim_new.model.load_state_dict(torch.load('qat_ckpt.pth'))
```

---

## 33. Version compatibility matrix (AIMET version vs PyTorch/TF version)
| AIMET Version | PyTorch Version | TensorFlow Version | ONNX Opset |
|---------------|-----------------|--------------------|------------|
| 1.24.x        | 1.13.1          | 2.10.x             | 11 - 13    |
| 1.25.x        | 2.0.1           | 2.12.x             | 13 - 15    |
| 1.26.x        | 2.1.0           | 2.14.x             | 14 - 17    |

---

## 34. Complete TensorFlow config file reference (separate from PyTorch)
The JSON configuration schema for TensorFlow is largely identical to PyTorch, but op_types refer to Keras layers (e.g., `Conv2D`, `Dense` instead of `Conv`, `Gemm`).
