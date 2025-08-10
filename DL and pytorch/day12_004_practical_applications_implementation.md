# Day 12.4: Practical Applications - A Project-Based Guide

## Introduction: Putting It All Together

Theory and isolated examples are essential, but the best way to solidify your understanding is to apply these concepts to a complete, practical project. This guide will walk you through a full, end-to-end project: **Time Series Forecasting of Airline Passenger Data**.

We will take a classic time series dataset, preprocess it, build an LSTM model to predict future passenger numbers, and visualize the results. This project will integrate many of the concepts we have learned, including data preparation, creating a `Dataset` and `DataLoader`, building a recurrent model, and training it effectively.

**Today's Learning Objectives:**

1.  **Perform Time Series Preprocessing:** Learn how to load, visualize, and scale time series data.
2.  **Apply the Sliding Window Technique:** Convert the time series into a supervised learning problem suitable for an LSTM.
3.  **Build a Complete `Dataset` and `DataLoader` Pipeline:** Encapsulate the data preparation logic into a reusable pipeline.
4.  **Construct and Train an LSTM Forecasting Model:** Build an LSTM model specifically for this regression task.
5.  **Evaluate and Visualize the Forecasts:** Compare the model's predictions against the true values and see how well it captures the trend and seasonality of the data.

---

## Part 1: The Airline Passengers Dataset

This is a classic dataset that contains the number of monthly international airline passengers from 1949 to 1960. It has a clear **trend** (passenger numbers are increasing over time) and **seasonality** (passenger numbers peak in the summer months).

### 1.1. Loading and Visualizing the Data

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

print("--- Part 1: Loading and Visualizing the Data ---")

# --- Load the data ---
# The data can be found in many places online.
# For reproducibility, we'll use a known URL.
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, usecols=[1], header=0)
data = df.values.astype(float)

# --- Visualize the data ---
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Monthly International Airline Passengers (1949-1960)')
plt.xlabel('Month')
plt.ylabel('Passengers (in 1000s)')
plt.grid(True)
plt.show()

# --- Preprocessing: Scaling ---
# LSTMs are sensitive to the scale of the input data. It's good practice to normalize
# or scale the data to a small range, like 0 to 1.
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

print(f"Dataset contains {len(data)} data points.")
```

---

## Part 2: Data Preparation with `Dataset` and `DataLoader`

We will now use the sliding window technique from Day 11 to create our input/output pairs and wrap it all in a PyTorch `Dataset`.

### 2.1. Creating the Sliding Windows

```python
def create_inout_sequences(input_data, tw):
    """Creates sliding window sequences."""
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# --- Define the window size ---
# We will use the last 12 months of data to predict the next month.
train_window = 12

# --- Split data into train and test ---
# We'll use the first 120 months for training and the rest for testing.
train_data = data_scaled[0:120]
test_data = data_scaled[120:]

train_inout_seq = create_inout_sequences(train_data, train_window)
test_inout_seq = create_inout_sequences(test_data, train_window)

print(f"\n--- Part 2: Data Preparation ---")
print(f"Created {len(train_inout_seq)} training sequences.")
print(f"Created {len(test_inout_seq)} testing sequences.")
```

### 2.2. Using a `DataLoader`

Since our data is already a list of `(sequence, label)` tuples, we can feed it directly to a `DataLoader`.

```python
from torch.utils.data import DataLoader

batch_size = 8
train_loader = DataLoader(train_inout_seq, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_inout_seq, batch_size=1, shuffle=False) # Batch size 1 for testing

print(f"DataLoader created with batch size {batch_size}.")
```

---

## Part 3: Building and Training the LSTM Model

### 3.1. The LSTM Forecaster Model

Our model will take a sequence of length 12 (with 1 feature) and output a single value for the next month's prediction.

```python
print("\n--- Part 3: Building and Training the Model ---")

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # The LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        
        # The output layer
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
        # We will initialize the hidden state in the forward pass

    def forward(self, input_seq):
        # Initialize hidden and cell states
        # The shape is (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        
        # Get LSTM outputs
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        
        # We only care about the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Pass through the linear layer
        predictions = self.linear(last_time_step_out)
        return predictions

# --- Instantiate model, loss, and optimizer ---
model = LSTMForecaster()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("LSTM model created and ready for training.")
```

### 3.2. The Training Loop

```python
epochs = 150

for i in range(epochs):
    model.train()
    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(seq)
        
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if (i+1) % 25 == 0:
        print(f'Epoch {i+1}/{epochs} Loss: {single_loss.item():.4f}')

print("\nFinished Training.")
```

---

## Part 4: Evaluation and Visualization

Now, let's use our trained model to make predictions on the test set and see how well it did.

```python
print("\n--- Part 4: Evaluation and Visualization ---")

model.eval()

test_predictions = []

with torch.no_grad():
    for seq, _ in test_loader:
        seq = seq.to(device)
        # The model expects (batch, seq_len, features), so we add the feature dim
        pred = model(seq)
        test_predictions.append(pred.item())

# --- Inverse transform to get actual passenger numbers ---
# We need to un-scale the predictions to compare them with the original data.
actual_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))

# --- Plot the results ---
# We need to align our predictions with the original data on the x-axis.
# The test data starts after the training data, and our first prediction is after the first window.
x_axis = np.arange(len(train_data) + train_window, len(train_data) + len(test_predictions) + train_window, 1)

plt.figure(figsize=(14, 7))
plt.title('Airline Passenger Forecast')
plt.ylabel('Passengers (in 1000s)')
plt.grid(True)

plt.plot(data, label='Original Data')
plt.plot(x_axis, actual_predictions, label='Model Predictions', color='red')

plt.axvline(x=len(train_data), c='green', linestyle='--', label='Train/Test Split')
plt.legend(loc='upper left')
plt.show()
```

**Interpreting the Plot:**
The plot shows the original data in blue. The green dashed line marks where the test set begins. The red line shows our model's predictions on the test set. A good model will produce a red line that closely follows the blue line in the test region, capturing both the upward trend and the seasonal peaks and valleys.

## Conclusion

This project demonstrates the power of LSTMs for a practical, real-world task. By correctly preprocessing the time series data into a supervised learning format and applying a standard LSTM architecture, we were able to build a model that can effectively learn and forecast complex temporal patterns.

**Key Project Takeaways:**

1.  **Data Prep is Everything:** The most critical step was using the sliding window technique to transform the time series into `(input, target)` pairs.
2.  **Scaling is Necessary:** LSTMs, like most neural networks, perform best when the input data is scaled to a small range (e.g., 0 to 1 or -1 to 1).
3.  **The `Dataset`/`DataLoader` Paradigm is Flexible:** The same data loading architecture we used for images and text works perfectly for our windowed time series data.
4.  **LSTMs Can Capture Trend and Seasonality:** The final plot shows that the LSTM was successful in learning both the long-term upward trend and the yearly seasonal cycle in the passenger data.

This end-to-end example provides a complete blueprint that you can adapt to a wide variety of other time series forecasting problems.

## Self-Assessment Questions

1.  **Sliding Window:** In our project, we used a `train_window` of 12. What is the real-world meaning of this number in the context of our monthly dataset?
2.  **Scaling:** Why did we need to apply `scaler.inverse_transform` to our predictions before plotting them?
3.  **Model Input/Output:** What was the exact shape of the input tensor that went into our `LSTMForecaster` model? What was the shape of the output?
4.  **Evaluation:** Why did we set `batch_size=1` and `shuffle=False` for our `test_loader`?
5.  **Model Improvement:** Looking at the final plot, what are some ways you might try to improve the model's performance even further?

