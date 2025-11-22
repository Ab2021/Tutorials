# Lab 03: Custom Autograd Function - ReLU from Scratch

## Difficulty
🟡 **Medium**

## Estimated Time
1 hour

## Learning Objectives
- Understand PyTorch's autograd mechanism deeply
- Implement custom forward and backward passes
- Master gradient computation for non-linear functions
- Debug gradient flow in neural networks
- Compare custom implementation with PyTorch's built-in functions

## Prerequisites
- Day 2: Autograd & Computational Graphs
- Understanding of derivatives and chain rule
- Familiarity with PyTorch tensors
- Basic knowledge of activation functions

## Problem Statement

Implement the ReLU (Rectified Linear Unit) activation function from scratch using PyTorch's `torch.autograd.Function`. Your implementation must:

1. Define the forward pass: `f(x) = max(0, x)`
2. Define the backward pass (gradient): `f'(x) = 1 if x > 0 else 0`
3. Handle edge cases (x = 0, negative values, gradients)
4. Support both scalar and tensor inputs
5. Integrate seamlessly with PyTorch's autograd system

### What is ReLU?

**ReLU (Rectified Linear Unit)** is one of the most popular activation functions in deep learning:

```
f(x) = max(0, x) = {
    x,  if x > 0
    0,  if x ≤ 0
}
```

**Derivative**:
```
f'(x) = {
    1,  if x > 0
    0,  if x ≤ 0
}
```

**Why ReLU?**
- Simple and computationally efficient
- Helps mitigate vanishing gradient problem
- Introduces non-linearity
- Sparse activation (many neurons output 0)

### Example

```python
Input: tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
Output: tensor([0.0, 0.0, 0.0, 1.0, 2.0])

Gradient (if output requires_grad):
Input gradient: tensor([1.0, 1.0, 1.0, 1.0, 1.0])
Output gradient: tensor([0.0, 0.0, 0.0, 1.0, 1.0])
```

## Requirements

1. Implement `ReLUFunction` class inheriting from `torch.autograd.Function`
2. Implement `forward()` static method
3. Implement `backward()` static method
4. Create a functional wrapper `custom_relu()`
5. Verify gradients using `torch.autograd.gradcheck()`
6. Compare with PyTorch's `F.relu()`
7. Handle in-place operations correctly

## Starter Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUFunction(torch.autograd.Function):
    """
    Custom ReLU activation function with manual gradient computation.
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: compute ReLU(x) = max(0, x)
        
        Args:
            ctx: Context object to save information for backward pass
            input: Input tensor
            
        Returns:
            Output tensor after ReLU activation
        """
        # TODO: Implement forward pass
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient of ReLU
        
        Args:
            ctx: Context object with saved information from forward pass
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        # TODO: Implement backward pass
        pass

def custom_relu(input):
    """
    Functional interface for custom ReLU.
    
    Args:
        input: Input tensor
        
    Returns:
        Output tensor after ReLU activation
    """
    return ReLUFunction.apply(input)

# Test code
def test_custom_relu():
    # Test forward pass
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = custom_relu(x)
    
    print(f"Input: {x}")
    print(f"Output: {y}")
    
    # Test backward pass
    loss = y.sum()
    loss.backward()
    
    print(f"Gradient: {x.grad}")
    
    # Compare with PyTorch's ReLU
    x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y2 = F.relu(x2)
    loss2 = y2.sum()
    loss2.backward()
    
    print(f"\\nPyTorch ReLU output: {y2}")
    print(f"PyTorch ReLU gradient: {x2.grad}")
    
    # Verify they match
    assert torch.allclose(y, y2), "Forward pass doesn't match!"
    assert torch.allclose(x.grad, x2.grad), "Backward pass doesn't match!"
    
    print("\\n✅ All tests passed!")

if __name__ == "__main__":
    test_custom_relu()
```

## Hints

<details>
<summary>Hint 1: Forward Pass</summary>

Use `torch.clamp()` or element-wise comparison:
```python
output = torch.clamp(input, min=0)
# OR
output = torch.where(input > 0, input, torch.zeros_like(input))
```

Don't forget to save information needed for backward pass using `ctx.save_for_backward()`!
</details>

<details>
<summary>Hint 2: Saving Context</summary>

You need to save the input (or a mask) to compute gradients in backward:
```python
ctx.save_for_backward(input)
# OR save just what you need
ctx.save_for_backward(input > 0)  # Boolean mask
```
</details>

<details>
<summary>Hint 3: Backward Pass</summary>

The gradient of ReLU is:
- 1 where input > 0
- 0 where input ≤ 0

Multiply `grad_output` by this mask:
```python
input, = ctx.saved_tensors
grad_input = grad_output * (input > 0).float()
```
</details>

<details>
<summary>Hint 4: Gradient Checking</summary>

Use PyTorch's gradient checker to verify correctness:
```python
from torch.autograd import gradcheck

input = torch.randn(10, dtype=torch.double, requires_grad=True)
test = gradcheck(custom_relu, input, eps=1e-6, atol=1e-4)
print(f"Gradient check: {test}")
```
</details>

## Solution

<details>
<summary>Click to reveal solution</summary>

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

class ReLUFunction(torch.autograd.Function):
    """
    Custom implementation of ReLU activation function.
    
    ReLU(x) = max(0, x)
    
    Gradient:
    ∂ReLU/∂x = 1 if x > 0, else 0
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: ReLU(x) = max(0, x)
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor of any shape
            
        Returns:
            Output tensor with same shape as input
        """
        # Save input for backward pass
        # We only need to know where input > 0
        ctx.save_for_backward(input)
        
        # Compute ReLU: max(0, x)
        output = input.clamp(min=0)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute ∂L/∂x given ∂L/∂y
        
        Chain rule: ∂L/∂x = ∂L/∂y * ∂y/∂x
        
        Where:
        - ∂L/∂y is grad_output (provided)
        - ∂y/∂x is the derivative of ReLU
        
        Args:
            ctx: Context with saved tensors
            grad_output: Gradient of loss w.r.t. output (∂L/∂y)
            
        Returns:
            Gradient of loss w.r.t. input (∂L/∂x)
        """
        # Retrieve saved input
        input, = ctx.saved_tensors
        
        # Create gradient mask: 1 where input > 0, else 0
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        
        # Alternative implementation:
        # grad_input = grad_output * (input > 0).float()
        
        return grad_input

def custom_relu(input):
    """
    Functional interface for custom ReLU.
    
    Args:
        input: Input tensor
        
    Returns:
        Output tensor after ReLU activation
    """
    return ReLUFunction.apply(input)

# Alternative implementation using torch.where
class ReLUFunctionAlt(torch.autograd.Function):
    """Alternative implementation using torch.where"""
    
    @staticmethod
    def forward(ctx, input):
        # Create mask for positive values
        mask = input > 0
        ctx.save_for_backward(mask)
        
        # Apply ReLU using where
        output = torch.where(mask, input, torch.zeros_like(input))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        
        # Gradient is grad_output where mask is True, else 0
        grad_input = torch.where(mask, grad_output, torch.zeros_like(grad_output))
        return grad_input

def custom_relu_alt(input):
    return ReLUFunctionAlt.apply(input)
```

### Detailed Walkthrough

Let's trace through an example step by step:

**Input**: `x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])`

**Forward Pass**:
```python
# Step 1: Save input for backward
ctx.save_for_backward(input)  # Save [-2, -1, 0, 1, 2]

# Step 2: Apply ReLU
output = input.clamp(min=0)   # [0, 0, 0, 1, 2]

# Visualization:
# input:  [-2, -1,  0,  1,  2]
#          ↓   ↓   ↓   ↓   ↓
# output: [ 0,  0,  0,  1,  2]
```

**Backward Pass** (assume `loss = output.sum()`):
```python
# grad_output from loss.backward()
grad_output = tensor([1.0, 1.0, 1.0, 1.0, 1.0])

# Step 1: Retrieve saved input
input = tensor([-2, -1, 0, 1, 2])

# Step 2: Create gradient mask
mask = (input > 0)  # [False, False, False, True, True]

# Step 3: Apply mask to grad_output
grad_input = grad_output * mask.float()
# = [1, 1, 1, 1, 1] * [0, 0, 0, 1, 1]
# = [0, 0, 0, 1, 1]

# Interpretation:
# - Gradients flow through where input > 0
# - Gradients are blocked where input ≤ 0
```

### Gradient Verification

```python
def verify_gradients():
    """Verify custom ReLU gradients using PyTorch's gradcheck."""
    
    # Use double precision for numerical gradient computation
    input = torch.randn(20, dtype=torch.double, requires_grad=True)
    
    # Check gradients
    test = gradcheck(custom_relu, input, eps=1e-6, atol=1e-4)
    
    if test:
        print("✅ Gradient check passed!")
    else:
        print("❌ Gradient check failed!")
    
    return test

verify_gradients()
```

### Comparison with PyTorch's ReLU

```python
def compare_with_pytorch():
    """Compare custom ReLU with PyTorch's implementation."""
    
    # Test data
    x_custom = torch.randn(1000, requires_grad=True)
    x_pytorch = x_custom.clone().detach().requires_grad_(True)
    
    # Forward pass
    y_custom = custom_relu(x_custom)
    y_pytorch = F.relu(x_pytorch)
    
    # Check forward pass
    assert torch.allclose(y_custom, y_pytorch), "Forward pass mismatch!"
    print("✅ Forward pass matches PyTorch")
    
    # Backward pass
    loss_custom = y_custom.sum()
    loss_pytorch = y_pytorch.sum()
    
    loss_custom.backward()
    loss_pytorch.backward()
    
    # Check gradients
    assert torch.allclose(x_custom.grad, x_pytorch.grad), "Gradient mismatch!"
    print("✅ Gradients match PyTorch")
    
    # Performance comparison
    import time
    
    x = torch.randn(10000, 10000)
    
    # Custom ReLU
    start = time.time()
    for _ in range(100):
        _ = custom_relu(x)
    custom_time = time.time() - start
    
    # PyTorch ReLU
    start = time.time()
    for _ in range(100):
        _ = F.relu(x)
    pytorch_time = time.time() - start
    
    print(f"\\nPerformance:")
    print(f"Custom ReLU: {custom_time:.4f}s")
    print(f"PyTorch ReLU: {pytorch_time:.4f}s")
    print(f"Slowdown: {custom_time/pytorch_time:.2f}x")

compare_with_pytorch()
```

### Edge Cases

```python
def test_edge_cases():
    """Test edge cases for custom ReLU."""
    
    # Test 1: Exactly zero
    x = torch.tensor([0.0], requires_grad=True)
    y = custom_relu(x)
    assert y.item() == 0.0, "ReLU(0) should be 0"
    
    y.backward()
    assert x.grad.item() == 0.0, "Gradient at 0 should be 0"
    print("✅ Zero input handled correctly")
    
    # Test 2: Very small positive
    x = torch.tensor([1e-10], requires_grad=True)
    y = custom_relu(x)
    assert y.item() > 0, "Small positive should pass through"
    
    y.backward()
    assert x.grad.item() == 1.0, "Gradient should be 1"
    print("✅ Small positive handled correctly")
    
    # Test 3: Very small negative
    x = torch.tensor([-1e-10], requires_grad=True)
    y = custom_relu(x)
    assert y.item() == 0.0, "Small negative should be 0"
    
    y.backward()
    assert x.grad.item() == 0.0, "Gradient should be 0"
    print("✅ Small negative handled correctly")
    
    # Test 4: Large values
    x = torch.tensor([1e10, -1e10], requires_grad=True)
    y = custom_relu(x)
    assert y[0].item() == 1e10, "Large positive preserved"
    assert y[1].item() == 0.0, "Large negative zeroed"
    print("✅ Large values handled correctly")
    
    # Test 5: Multidimensional
    x = torch.randn(10, 20, 30, requires_grad=True)
    y = custom_relu(x)
    assert y.shape == x.shape, "Shape should be preserved"
    
    loss = y.sum()
    loss.backward()
    assert x.grad.shape == x.shape, "Gradient shape should match input"
    print("✅ Multidimensional tensors handled correctly")

test_edge_cases()
```

### Using in a Neural Network

```python
class CustomReLUNet(nn.Module):
    """Neural network using custom ReLU activation."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = custom_relu(x)  # Use custom ReLU
        x = self.fc2(x)
        return x

# Test the network
model = CustomReLUNet(10, 20, 2)
x = torch.randn(5, 10)
y = model(x)

# Verify gradients flow correctly
loss = y.sum()
loss.backward()

print("✅ Custom ReLU works in neural network!")
```

</details>

## Test Cases

```python
def comprehensive_tests():
    """Comprehensive test suite for custom ReLU."""
    
    print("Running comprehensive tests...\\n")
    
    # Test 1: Basic functionality
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = custom_relu(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    assert torch.allclose(y, expected), "Basic forward pass failed"
    print("✅ Test 1: Basic functionality")
    
    # Test 2: Gradient computation
    loss = y.sum()
    loss.backward()
    expected_grad = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(x.grad, expected_grad), "Gradient computation failed"
    print("✅ Test 2: Gradient computation")
    
    # Test 3: Gradient check
    x = torch.randn(10, dtype=torch.double, requires_grad=True)
    assert gradcheck(custom_relu, x, eps=1e-6, atol=1e-4), "Gradient check failed"
    print("✅ Test 3: Gradient check")
    
    # Test 4: Match PyTorch
    x1 = torch.randn(100, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)
    
    y1 = custom_relu(x1)
    y2 = F.relu(x2)
    assert torch.allclose(y1, y2), "Forward doesn't match PyTorch"
    
    y1.sum().backward()
    y2.sum().backward()
    assert torch.allclose(x1.grad, x2.grad), "Gradient doesn't match PyTorch"
    print("✅ Test 4: Matches PyTorch")
    
    # Test 5: Multidimensional
    x = torch.randn(10, 20, 30, requires_grad=True)
    y = custom_relu(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad.shape == x.shape
    print("✅ Test 5: Multidimensional tensors")
    
    print("\\n🎉 All comprehensive tests passed!")

comprehensive_tests()
```

## Extensions

1. **Leaky ReLU**: Implement Leaky ReLU where `f(x) = x if x > 0 else α*x` (α is a small constant like 0.01)

2. **Parametric ReLU**: Make α a learnable parameter

3. **ELU (Exponential Linear Unit)**: Implement `f(x) = x if x > 0 else α*(e^x - 1)`

4. **GELU**: Implement Gaussian Error Linear Unit (used in transformers)

5. **In-place ReLU**: Implement an in-place version that modifies the input tensor directly

6. **Benchmarking**: Compare performance of custom vs PyTorch ReLU on GPU

## Related Concepts
- [Day 2: Autograd & Computational Graphs](../Day2_Autograd.md)
- [Lab 04: Custom Loss Function](lab_04_custom_loss.md)
- [Lab 05: Gradient Clipping](lab_05_gradient_clipping.md)

## Real-World Applications

1. **Custom Activation Functions**: Research new activation functions
2. **Quantization**: Implement quantized versions of activations
3. **Gradient Analysis**: Debug gradient flow in complex networks
4. **Hardware Acceleration**: Optimize for specific hardware (TPU, custom ASICs)
5. **Differentiable Algorithms**: Implement custom differentiable operations

## Key Takeaways

1. **Autograd Function**: Use `torch.autograd.Function` for custom operations
2. **Context Object**: Save tensors needed for backward pass using `ctx.save_for_backward()`
3. **Gradient Computation**: Manually compute ∂L/∂x = ∂L/∂y * ∂y/∂x
4. **Gradient Checking**: Always verify with `gradcheck()` for numerical stability
5. **Performance**: PyTorch's built-in functions are optimized; custom functions are for research/special cases

## Common Pitfalls

1. **Forgetting to save context**: Not calling `ctx.save_for_backward()`
2. **Wrong gradient shape**: Ensure `grad_input` has same shape as `input`
3. **Not handling edge cases**: Special handling for x=0, NaN, inf
4. **In-place operations**: Modifying saved tensors breaks autograd
5. **Double precision**: Use `dtype=torch.double` for gradient checking

---

**Next**: [Lab 04: Custom Loss Function](lab_04_custom_loss.md)
