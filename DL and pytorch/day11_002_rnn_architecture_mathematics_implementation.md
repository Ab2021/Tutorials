# Day 11.2: RNN Architecture & Mathematics - A Practical Guide

## Introduction: Unrolling the Recurrence

We have learned that a Recurrent Neural Network (RNN) processes sequences by maintaining a hidden state that acts as its memory. But what is the actual math happening inside that recurrent loop? How do the weights, inputs, and previous hidden state combine to produce the new hidden state?

This guide will provide a deep dive into the mathematics of a simple RNN cell. We will build it from scratch again, but this time with a focus on the underlying matrix operations and the flow of information. We will also "unroll" the network through time to visualize how the same set of weights is applied at every step, which is crucial for understanding how backpropagation works in an RNN.

**Today's Learning Objectives:**

1.  **Understand the Core RNN Equations:** See the mathematical formulas that define the transformation from one time step to the next.
2.  **Implement the RNN Cell with Matrix Operations:** Build an RNN cell using `nn.Linear` layers to clearly see the role of the weight matrices.
3.  **Visualize the Unrolled RNN:** Grasp the concept of unrolling the temporal loop into a deep feed-forward network, which clarifies the concepts of parameter sharing and gradient flow.
4.  **Connect Theory to Practice:** Explicitly map the mathematical equations to the corresponding PyTorch code.

---

## Part 1: The Mathematics of a Simple RNN Cell

At any given time step `t`, a simple RNN cell (often called an Elman network) performs two calculations:

1.  **The Hidden State Update:**

    `h_t = f_h(W_hh * h_{t-1} + W_xh * x_t + b_h)`

    *   `h_t`: The new hidden state at time `t`. This is a vector of size `hidden_size`.
    *   `h_{t-1}`: The hidden state from the previous time step (`t-1`). Also a vector of size `hidden_size`.
    *   `x_t`: The input vector for the current time step `t`. This is a vector of size `input_size`.
    *   `W_hh`: The **hidden-to-hidden weight matrix**. It transforms the previous hidden state. Shape: `(hidden_size, hidden_size)`.
    *   `W_xh`: The **input-to-hidden weight matrix**. It transforms the current input. Shape: `(hidden_size, input_size)`.
    *   `b_h`: The bias for the hidden state calculation. A vector of size `hidden_size`.
    *   `f_h`: The hidden layer activation function, typically `tanh` or `ReLU`.

2.  **The Output Calculation (Optional):**

    `y_t = f_y(W_hy * h_t + b_y)`

    *   `y_t`: The output for the current time step `t`. A vector of size `output_size`.
    *   `W_hy`: The **hidden-to-output weight matrix**. It transforms the current hidden state to produce the output. Shape: `(output_size, hidden_size)`.
    *   `b_y`: The bias for the output calculation. A vector of size `output_size`.
    *   `f_y`: The output layer activation function (e.g., `Softmax` for classification).

**Key Insight: Parameter Sharing**
The most important thing to realize is that the weight matrices `W_hh`, `W_xh`, and `W_hy` are the **same** for every single time step. The model learns one set of weights and applies it over and over again in the recurrent loop. This is what makes RNNs so parameter-efficient.

---

## Part 2: Implementing the Math in PyTorch

Let's translate these equations directly into a PyTorch `nn.Module`.

```python
import torch
import torch.nn as nn

print("---" + " Part 2: Implementing the RNN Math " + "---")

class ManualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManualRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # --- Define the learnable weight matrices and biases ---
        # Corresponds to W_xh and its bias
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        
        # Corresponds to W_hh and its bias
        # Note: Often the bias is combined, but we separate it for clarity.
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # The activation function
        self.activation = nn.Tanh()

    def forward(self, x_t, h_t_minus_1):
        """Performs a single time step."""
        # --- Apply the core RNN equation ---
        # 1. Transform the input
        transformed_input = self.input_to_hidden(x_t)
        
        # 2. Transform the previous hidden state
        transformed_hidden = self.hidden_to_hidden(h_t_minus_1)
        
        # 3. Sum them up (this is the `+ b_h` part, as biases are included in nn.Linear)
        combined = transformed_input + transformed_hidden
        
        # 4. Apply the non-linear activation function
        h_t = self.activation(combined)
        
        return h_t

# --- Usage Example ---
# Parameters
input_size = 10
hidden_size = 20
batch_size = 4

# Create the cell
rnn_cell = ManualRNNCell(input_size, hidden_size)

# Create dummy data for a single time step
input_t = torch.randn(batch_size, input_size)
prev_hidden_state = torch.randn(batch_size, hidden_size)

# Calculate the new hidden state
new_hidden_state = rnn_cell(input_t, prev_hidden_state)

print(f"Input shape (x_t): {input_t.shape}")
print(f"Previous hidden state shape (h_{{t-1}}): {prev_hidden_state.shape}")
print(f"New hidden state shape (h_t): {new_hidden_state.shape}")
```

---

## Part 3: Unrolling the RNN Through Time

To understand how gradients flow in an RNN (Backpropagation Through Time - BPTT), it's helpful to visualize the network as being **unrolled**. Unrolling means creating a separate copy of the network for each time step in the sequence. The hidden state of one copy is then fed as input to the next.

This transforms the recurrent network into a very deep feed-forward network, where each layer corresponds to a time step. Crucially, all these unrolled layers **share the exact same weights**.

**Sequence:** `[x_0, x_1, x_2]`

**Unrolled Computation:**

*   `h_0 = f_h(W_hh * h_{init} + W_xh * x_0)`
*   `h_1 = f_h(W_hh * h_0 + W_xh * x_1)`
*   `h_2 = f_h(W_hh * h_1 + W_xh * x_2)`

![Unrolled RNN](https://i.imgur.com/P4iY1iI.png)

**Why this is important:**

*   **Gradient Flow:** When we calculate the loss at the end of the sequence (or at every step) and call `.backward()`, the gradients must flow backward through this entire unrolled structure. The gradient for `W_hh` at time step `t=0` is the sum of the gradients flowing back from `t=0`, `t=1`, and `t=2`.
*   **Vanishing/Exploding Gradients:** Look at the path from `h_2` back to `h_0`. The gradient has to pass through the `W_hh` matrix multiple times. If the eigenvalues of `W_hh` are small, the gradient will shrink exponentially as it flows back, leading to **vanishing gradients**. If they are large, it will grow exponentially, leading to **exploding gradients**. This is the fundamental challenge of training RNNs, which we will address in the next guide.

### 3.1. A Full Sequence Processor using the Manual Cell

Let's write the code that explicitly performs this unrolling.

```python
print("\n---" + " Part 3: Unrolling the RNN " + "---")

class FullRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # The single RNN cell that will be applied at every time step
        self.rnn_cell = ManualRNNCell(input_size, hidden_size)
        
        # A final layer to map the hidden state to an output
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        """
        Processes an entire sequence by unrolling the RNN cell.
        Args:
            input_sequence: Tensor of shape (batch_size, seq_len, input_size)
        """
        batch_size = input_sequence.size(0)
        seq_len = input_sequence.size(1)
        
        # Initialize the hidden state
        h_t = torch.zeros(batch_size, self.hidden_size)
        
        # List to store the outputs at each time step
        outputs = []
        
        # This loop is the unrolling process
        for t in range(seq_len):
            # Get the input for this time step
            x_t = input_sequence[:, t, :]
            
            # Apply the SAME rnn_cell
            h_t = self.rnn_cell(x_t, h_t)
            
            # Calculate the output for this step
            output_t = self.hidden_to_output(h_t)
            outputs.append(output_t)
            
        # Stack the outputs
        # The shape will be (batch_size, seq_len, output_size)
        return torch.stack(outputs, dim=1)

# --- Usage Example ---
# Parameters
input_size = 10
hidden_size = 20
output_size = 5
seq_len = 7
batch_size = 4

# Create the full RNN model
full_rnn_model = FullRNN(input_size, hidden_size, output_size)

# Create a dummy input sequence
sequence = torch.randn(batch_size, seq_len, input_size)

# Get the outputs for the entire sequence
all_outputs = full_rnn_model(sequence)

print(f"Input sequence shape: {sequence.shape}")
print(f"Final output sequence shape: {all_outputs.shape}")
```

## Conclusion

By breaking down the RNN into its core mathematical equations and implementing them from scratch, we can clearly see the mechanism of recurrence. The concept of unrolling the network through time is the key to understanding both its strengths (parameter sharing) and its weaknesses (the difficulty of propagating gradients over long sequences).

**Key Takeaways:**

1.  **Core Equation:** The new hidden state `h_t` is a non-linear function of a combination of the previous hidden state `h_{t-1}` and the current input `x_t`.
2.  **Weight Matrices:** An RNN has two primary weight matrices, `W_xh` (input-to-hidden) and `W_hh` (hidden-to-hidden), that are shared across all time steps.
3.  **Unrolling:** A recurrent network processing a sequence of length `T` can be viewed as a `T`-layer deep feed-forward network where all layers share the same weights.
4.  **Gradient Flow:** This unrolled view makes it clear that gradients must travel back through the entire sequence, repeatedly being multiplied by the `W_hh` matrix, which is the source of the vanishing and exploding gradient problems.

This mathematical foundation is essential for appreciating why more complex architectures like LSTMs and GRUs, which we will explore next, were necessary inventions.

## Self-Assessment Questions

1.  **Parameter Sharing:** In the core RNN equation, which components are shared across all time steps?
2.  **Matrix Shapes:** If your `input_size` is 50 and your `hidden_size` is 100, what are the shapes of the `W_xh` and `W_hh` weight matrices?
3.  **Unrolling:** If you unroll an RNN for a sequence of 20 time steps, how many unique `W_hh` matrices are there in the resulting computation graph?
4.  **The `tanh` function:** What is the purpose of the `tanh` activation function in the hidden state update equation?
5.  **Gradient Flow:** When backpropagating, the gradient of the loss with respect to `h_{t-1}` depends on the gradient with respect to `h_t`. What weight matrix is involved in this specific step of the backpropagation chain rule?

```