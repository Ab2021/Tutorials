# Day 1.4: Business Context & Problem Formulation - A Practical Guide

## Introduction: From Business Goals to Machine Learning Models

This is one of the most critical and often overlooked steps in any data science project. A model that is 99% accurate is useless if it solves the wrong problem. Before writing a single line of code, we must translate a vague business goal into a precise, solvable machine learning problem.

This guide will walk you through this translation process using a series of practical case studies. We will focus on three key questions:

1.  **What is the business problem?** (The high-level goal)
2.  **How can we frame this as a machine learning problem?** (The specific task: classification, regression, etc.)
3.  **How will we measure success?** (The evaluation metrics that align with the business goal)

For each case study, we will define the problem and then sketch out a simple PyTorch implementation to show how the problem formulation directly influences the model's architecture, its output, and the loss function we choose.

**Today's Learning Objectives:**

1.  **Learn a structured framework** for converting business needs into ML problem statements.
2.  **Analyze three distinct case studies:** Customer Churn, Predictive Maintenance, and Customer Lifetime Value (CLV) Prediction.
3.  **Understand the mapping** between problem type (classification, regression) and PyTorch components (`nn.BCELoss`, `nn.MSELoss`, model output layers).
4.  **Appreciate the importance of choosing the right evaluation metric** (Accuracy vs. Precision/Recall vs. MAE/RMSE) based on the business impact of errors.

---

## Case Study 1: Customer Churn Prediction

### 1.1. The Business Problem

**The Scenario:** A telecom company is losing customers to competitors. It costs five times more to acquire a new customer than to retain an existing one. The business goal is to **reduce customer churn** by proactively identifying at-risk customers and offering them incentives to stay.

**The Vague Goal:** "Let's stop customers from leaving."

### 1.2. Framing the Machine Learning Problem

We need to make this goal specific and actionable.

*   **The Question:** For a given customer, will they churn (cancel their subscription) within the next month?
*   **The ML Framing:** This is a **binary classification** problem. The two classes are "Churn" (1) and "No Churn" (0).

*   **Input Data (Features):** What do we know about our customers? We need to gather relevant data.
    *   *Demographics:* Age, gender, tenure (how long they've been a customer).
    *   *Service Usage:* Monthly charges, total charges, number of services subscribed to (phone, internet, TV, etc.).
    *   *Contract Information:* Type of contract (month-to-month, one-year, two-year), payment method.
    *   *Customer Service Interactions:* Number of support tickets, etc.

*   **Output Target:** A single value for each customer:
    *   `0`: The customer did not churn.
    *   `1`: The customer did churn.

### 1.3. Choosing the Right Evaluation Metric

**Why Accuracy is Not Enough:** Imagine only 3% of customers churn each month. A model that predicts "No Churn" for everyone would be 97% accurate, but completely useless for our business goal! We need to identify the churners.

*   **False Positives (FP):** We predict a customer will churn, but they don't. *Business Cost:* We waste money on an incentive for a happy customer.
*   **False Negatives (FN):** We predict a customer will *not* churn, but they do. *Business Cost:* We lose a customer, which is very expensive.

Clearly, the cost of a False Negative is much higher than the cost of a False Positive. Therefore, we should prioritize metrics that are sensitive to the minority class (Churn).

*   **Good Metrics for this Problem:**
    *   **Recall (Sensitivity):** Of all the customers who *actually* churned, what percentage did our model correctly identify? `Recall = TP / (TP + FN)`. We want to maximize this.
    *   **Precision:** Of all the customers we *predicted* would churn, what percentage actually did? `Precision = TP / (TP + FP)`. We also want this to be high to avoid wasting incentives.
    *   **F1-Score:** The harmonic mean of Precision and Recall. A good single metric to balance the two.
    *   **AUC (Area Under the ROC Curve):** A great overall measure of a classifier's ability to distinguish between the two classes.

### 1.4. PyTorch Implementation Sketch

This sketch shows how the problem formulation dictates the model's structure.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. Create a Dummy Dataset ---
# In a real project, this would come from a database/CSV file.
class ChurnDataset(Dataset):
    def __init__(self, num_samples=1000, num_features=10):
        # Generate some random data to simulate customer features
        self.X = torch.randn(num_samples, num_features)
        # Generate random labels (0 or 1), with churn being less frequent (e.g., 10% churn rate)
        self.y = (torch.rand(num_samples) < 0.1).float().view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. Define the Model ---
# A simple Multi-Layer Perceptron (MLP) for tabular data.
class ChurnPredictor(nn.Module):
    def __init__(self, input_features):
        super(ChurnPredictor, self).__init__()
        self.layer1 = nn.Linear(input_features, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1) # CRITICAL: Output is 1 neuron

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        # CRITICAL: We pass the final output through a sigmoid function.
        # This squashes the output to a probability between 0 and 1.
        x = torch.sigmoid(self.output_layer(x))
        return x

# --- 3. Training Setup ---
# Instantiate dataset and dataloader
train_dataset = ChurnDataset()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Instantiate model
# The number of input features must match our data (10 in this dummy case)
model = ChurnPredictor(input_features=10)

# CRITICAL: For binary classification, we use Binary Cross-Entropy (BCE) Loss.
# This loss function is designed to work with probabilities from a sigmoid output.
loss_function = nn.BCELoss()

# Use a standard optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("--- Churn Prediction (Binary Classification) Setup ---")
print(f"Model Output Layer: {model.output_layer}")
print(f"Activation on Output: Sigmoid")
print(f"Loss Function: {type(loss_function).__name__}")

# The training loop would then proceed as usual, feeding the model batches of
# customer data and training it to predict the probability of churn.
```

---

## Case Study 2: Predictive Maintenance

### 2.1. The Business Problem

**The Scenario:** A manufacturing company operates hundreds of critical machines. When a machine fails unexpectedly, it causes production to halt, leading to significant financial losses. The business goal is to **minimize downtime** by predicting machine failures *before* they happen.

**The Vague Goal:** "Let's prevent machines from breaking down."

### 2.2. Framing the Machine Learning Problem

*   **The Question:** For a given machine, how many hours are left until its next failure?
*   **The ML Framing:** This is a **regression** problem. We are predicting a continuous numerical value (time until failure).

*   **Input Data (Features):** We need time-series data from sensors on the machines.
    *   *Vibration readings*
    *   *Temperature*
    *   *Rotational speed (RPM)*
    *   *Power consumption*
    *   *Machine age, model, last service date*

*   **Output Target:** A single number for each machine at a given time: the remaining useful life (RUL) in hours.

### 2.3. Choosing the Right Evaluation Metric

**Why Accuracy is Irrelevant:** Accuracy is a classification metric. For regression, we need to measure how close our predictions are to the true values.

*   **Over-prediction:** We predict a failure will happen in 10 hours, but it actually happens in 50 hours. *Business Cost:* We perform maintenance too early, which is inefficient but not catastrophic.
*   **Under-prediction:** We predict a failure will happen in 100 hours, but it actually happens in 20 hours. *Business Cost:* The machine fails unexpectedly, causing a production shutdown. This is the disaster we want to avoid.

We care more about large under-predictions. The penalty for being wrong should increase as the error gets larger.

*   **Good Metrics for this Problem:**
    *   **Mean Absolute Error (MAE):** `(1/n) * sum(|y_true - y_pred|)`. Simple to interpret (e.g., "On average, we are off by 10 hours").
    *   **Root Mean Squared Error (RMSE):** `sqrt((1/n) * sum((y_true - y_pred)^2))`. This is the most common and generally the best metric here. By squaring the error, it penalizes large errors (like our disastrous under-predictions) much more heavily than MAE.

### 2.4. PyTorch Implementation Sketch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Dummy Data ---
# For this problem, the input would likely be a sequence of sensor readings.
# We might use an RNN or a 1D CNN, but for simplicity, we'll use an MLP again.
num_samples = 500
num_features = 15 # e.g., summary statistics of sensor readings over the last hour
X_train = torch.randn(num_samples, num_features)
# The target is a continuous value (Remaining Useful Life in hours)
y_train = torch.rand(num_samples, 1) * 100 # RUL from 0 to 100 hours

# --- 2. Define the Model ---
class RULPredictor(nn.Module):
    def __init__(self, input_features):
        super(RULPredictor, self).__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.layer2 = nn.Linear(64, 32)
        # CRITICAL: Output is 1 neuron for the RUL value.
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        # CRITICAL: There is NO activation function on the output.
        # We want to predict any positive number, not a probability.
        x = self.output_layer(x)
        return x

# --- 3. Training Setup ---
model = RULPredictor(input_features=15)

# CRITICAL: For regression, we use Mean Squared Error (MSE) Loss.
# Minimizing MSE is equivalent to minimizing RMSE.
loss_function = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Predictive Maintenance (Regression) Setup ---")
print(f"Model Output Layer: {model.output_layer}")
print(f"Activation on Output: None")
print(f"Loss Function: {type(loss_function).__name__}")
```

---

## Case Study 3: Customer Lifetime Value (CLV) Prediction

### 2.1. The Business Problem

**The Scenario:** An e-commerce company wants to identify its most valuable customers to create targeted marketing campaigns and VIP programs. The business goal is to **maximize marketing return on investment (ROI)** by focusing efforts on high-value customers.

**The Vague Goal:** "Let's find our best customers."

### 2.2. Framing the Machine Learning Problem

*   **The Question:** What is the total amount of revenue a customer will generate over the next year?
*   **The ML Framing:** This is also a **regression** problem, similar to predictive maintenance, as we are predicting a continuous value (money).

*   **Input Data (Features):** We need historical transaction and engagement data.
    *   *Recency:* How recently did the customer make a purchase?
    *   *Frequency:* How often do they make purchases?
    *   *Monetary Value:* What is the average value of their purchases?
    *   *Demographics, products viewed, time spent on site, etc.*

*   **Output Target:** A single number: the predicted total spend for the next 365 days.

### 2.3. Choosing the Right Evaluation Metric

This is similar to the predictive maintenance case. We want to know how close our monetary predictions are to the actual future revenue.

*   **Good Metrics for this Problem:**
    *   **Mean Absolute Error (MAE):** Very interpretable in this context. "Our CLV predictions are off by an average of $50." This is often the primary metric.
    *   **Root Mean Squared Error (RMSE):** Also good, as it will heavily penalize being spectacularly wrong about a high-value customer.

### 2.4. PyTorch Implementation Sketch

The implementation for this would be **almost identical to the predictive maintenance case**. This is a key insight: many different business problems map to the same underlying ML problem type.

The only things that would change are:
1.  The **input features** would be different (Recency, Frequency, Monetary instead of sensor data).
2.  The **interpretation** of the output and the loss would be in dollars instead of hours.

The model architecture (`CLVPredictor`), the loss function (`nn.MSELoss`), and the lack of an output activation function would all be the same.

## Conclusion: A Framework for Success

We have seen how to take a vague business objective and systematically break it down into a concrete machine learning plan. This process is paramount to success.

**The Framework:**

1.  **Understand the Business Goal:** What does the business want to achieve? (e.g., reduce costs, increase revenue).
2.  **Frame the ML Problem:** Is it binary classification, multi-class classification, or regression? What are you trying to predict?
3.  **Define Inputs and Outputs:** What data do you have (features)? What is the exact definition of your target variable?
4.  **Select an Appropriate Metric:** How will you measure success? Choose a metric that aligns with the business cost of being wrong.
5.  **Translate to Code:** The problem framing directly determines your model's output layer, the final activation function, and the loss function.

| ML Problem Type         | Model Output (# Neurons) | Final Activation | Loss Function   |
|-------------------------|--------------------------|------------------|-----------------|
| Binary Classification   | 1                        | `torch.sigmoid`  | `nn.BCELoss`    |
| Multi-Class Classification | `num_classes`            | `nn.Softmax` (often built into loss) | `nn.CrossEntropyLoss` |
| Regression              | 1                        | None             | `nn.MSELoss`    |

By following this structured approach, you ensure that the model you build is not just technically correct, but that it provides real, measurable value to the business.

## Self-Assessment Questions

1.  **Problem Framing:** Your company wants to identify fraudulent credit card transactions. How would you frame this as an ML problem? (Classification or regression? What are the classes?)
2.  **Metrics:** For the fraud detection problem, what would be the business cost of a False Positive (flagging a real transaction as fraud)? What about a False Negative (letting a fraudulent transaction go through)? Which metric (Precision or Recall) would be more important to maximize?
3.  **Output Layers:** You are building a model to predict which of 50 different product categories a user is most likely to be interested in. What should the size of your model's output layer be? What loss function would you use?
4.  **Activation Functions:** Why do we not use an activation function on the output layer for regression problems?
5.  **Business to ML:** A real estate company wants to predict house prices based on features like square footage, number of bedrooms, and location. Walk through the 5 steps of the framework to define this problem.
