# Lab 4: Custom Autograd Function

## Objective
Understand how `autograd` works by implementing a custom Function.
We will implement a **ReLU** function manually.

## 1. The Function (`custom_relu.py`)

```python
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # Gradient of ReLU is 1 if input > 0, else 0
        grad_input[input < 0] = 0
        return grad_input

# 2. Test it
x = torch.tensor([-1.0, 1.0, 2.0], requires_grad=True)
relu = MyReLU.apply

y = relu(x)
z = y.sum()
z.backward()

print(f"Input: {x}")
print(f"Output: {y}")
print(f"Grad: {x.grad}") # Should be [0, 1, 1]
```

## 2. Analysis
Why do we need custom functions?
*   Non-differentiable operations (e.g., quantization).
*   Performance optimization (fused kernels).
*   Numerical stability fixes.

## 3. Challenge
Implement a **Sigmoid** function: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
Gradient: $\sigma(x) * (1 - \sigma(x))$.

## 4. Submission
Submit the code for `MySigmoid`.
