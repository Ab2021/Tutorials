# Day 137: Trust but Verify: Model Testing Strategy
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 20: CI/CD for ML

---

> **🎯 Focus Area:** Your model compiles. It runs. But does it work? **ML Testing** combines traditional Software Testing (Unit/Integration) with Statistical Testing (Data/Model Quality).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Write** Unit Tests for custom layers and loss functions using `pytest`.
2.  **Implement** "Minimal Functionality Tests" (Overfit on a single batch).
3.  **Create** Invariance Tests (Rotating an image shouldn't change the label).
4.  **Define** Directional Expectation Tests (Adding noise should increase entropy).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install pytest torch`.

---

## 📖 Theoretical Foundation

### 1. The Pyramid of ML Testing
*   **Infrastructure Tests:** Can I provision the GPU? (Terraform/Platform).
*   **Code Tests:** Is `loss.backward()` actually updating weights? (Unit).
*   **Data Tests:** Are there Nulls? Is Age < 0? (Great Expectations).
*   **Model Tests:** Does `Sentiment("I hate this")` return Negative? (Behavioral).

### 2. Behavioral Testing (CheckList)
*   **Invariance:** $f(x) == f(Augment(x))$.
*   **Directionality:** If $HouseSize$ increases, $Price$ should increase.
*   **Minimum Performance:** Accuracy > Random Guessing (0.5).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Code Unit Tests

Testing the *implementation* of the math.

#### 📁 `tests/test_model_code.py`
```python
import torch
import pytest
from src.model import MyCustomLayer

def test_layer_shape():
    # 1. Arrange
    layer = MyCustomLayer(in_features=10, out_features=5)
    x = torch.randn(32, 10)
    
    # 2. Act
    y = layer(x)
    
    # 3. Assert
    assert y.shape == (32, 5)

def test_gradient_flow():
    # Verify we didn't accidentally detach() gradients
    layer = MyCustomLayer(10, 5)
    x = torch.randn(32, 10)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Assert gradients exist and are non-zero
    assert layer.weights.grad is not None
    assert torch.abs(layer.weights.grad).sum() > 0.0
```

### 👨‍💻 Core Implementation: Minimal Functionality Test (Overfit)

If your model cannot memorize 10 examples, it implies a bug in the code (Learning rate too high/low, bad labels, broken data loader).

#### 📁 `tests/test_convergence.py`
```python
def test_overfit_small_batch():
    # 1. Create tiny dataset
    X = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # 2. Train for 100 steps
    for _ in range(100):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        
    # 3. Assert Loss is near zero
    assert loss.item() < 0.01, f"Failed to overfit! metrics: {loss.item()}"
```

### 👨‍💻 Core Implementation: Behavioral Tests

#### 📁 `tests/test_behavior.py`
```python
def test_sentiment_negation():
    model = load_best_model()
    
    text1 = "The movie was good."
    text2 = "The movie was not good."
    
    score1 = model.predict(text1)
    score2 = model.predict(text2)
    
    # Expect score1 > score2 (Directional)
    assert score1 > score2, "Negation should lower sentiment score"

def test_invariance_uppercase():
    model = load_best_model()
    text = "The fast car"
    
    assert model.predict(text) == model.predict(text.upper())
```

---

## 🔬 Lab Exercise: "The Determinism Trap"

### Task
Fix Flaky Tests.
1.  Run the tests. They pass.
2.  Run again. `test_overfit_small_batch` fails.
3.  **Cause:** Random initialization of weights. Occasionally it gets stuck in local minima or LR is edge case.
4.  **Fix:** Seed Everything.
    ```python
    @pytest.fixture(autouse=True)
    def set_seed():
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
    ```

---

## 📝 Daily Summary

### Key Takeaways
1.  **Don't Mock Math:** Unit testing stochastic functions (like SGD) is hard. Use "Property Based Testing" (outputs are within range [0,1]) rather than exact equality.
2.  **Regression Testing:** Keep a "Golden Set" of inputs. If the model prediction changes significantly on this set, trigger a warning.
3.  **Speed:** Code tests run in CI (Minutes). Training tests run in CD (Hours).

### API Summary
```python
pytest.approx(value, rel=1e-3)
```

---

**Day 137 Complete** ✅

*Next: Day 138 - Continuous Deployment (CD) - Canary vs Shadow.*
