# Day 27: Testing ML Systems

> **Phase**: 3 - System Design
> **Week**: 6 - Data Engineering
> **Focus**: QA for AI
> **Reading Time**: 45 mins

---

## 1. The Testing Pyramid for ML

ML systems need more than just unit tests.

### 1.1 Data Tests
*   **Schema Validation**: Are columns present? Are types correct?
*   **Value Validation**: Is `age` between 0 and 120? Is `probability` between 0 and 1?
*   **Tool**: Great Expectations, Pandera.

### 1.2 Unit Tests
*   **Code Logic**: Does `preprocess_text("Hello")` return `["hello"]`?
*   **Model Config**: Does the model architecture match the config? (e.g., correct output shape).

### 1.3 Model Tests (Behavioral)
*   **Invariance**: `predict("I love this movie")` should equal `predict("I love this film")`.
*   **Directionality**: `predict("House with 2 rooms")` price should be < `predict("House with 3 rooms")`.
*   **Minimum Functionality**: Does it perform better than random guessing?

---

## 2. Real-World Challenges & Solutions

### Challenge 1: Stochasticity
**Scenario**: `train_model()` produces different results every time. Unit tests fail.
**Solution**:
*   **Fixed Seeds**: Set `random.seed(42)` in tests.
*   **Approximate Assertions**: `assert 0.8 < accuracy < 0.9` instead of `accuracy == 0.85`.

### Challenge 2: Integration Testing
**Scenario**: The pipeline runs for 5 hours. You can't run this on every commit.
**Solution**:
*   **Tiny Dataset**: Create a "fixture" dataset with 100 rows that covers edge cases. Run the full pipeline on this tiny data in CI/CD.

---

## 3. Interview Preparation

### Conceptual Questions

**Q1: How do you test a model before deployment without ground truth?**
> **Answer**:
> *   **Slicing**: Check performance on critical subgroups (e.g., "Users in US", "Users on Mobile"). Ensure no subgroup is disproportionately harmed.
> *   **Consistency Checks**: Check invariance (synonyms) and directional expectations.
> *   **Shadow Mode**: Compare output distribution with the production model.

**Q2: What is a "Golden Set"?**
> **Answer**: A curated dataset of examples with verified ground truth labels, covering known edge cases and critical scenarios. The model *must* pass the Golden Set with high accuracy to be deployed. It acts as a regression test suite.

**Q3: Explain "Differential Testing".**
> **Answer**: Compare the outputs of the new model version against the old version on the same inputs. If the outputs differ significantly for 20% of users, investigate why. It helps detect unintended behavioral changes.

---

## 5. Further Reading
- [Testing Machine Learning Systems (Google)](https://developers.google.com/machine-learning/testing-debugging)
- [CheckList: Behavioral Testing of NLP Models](https://github.com/marcotcr/checklist)
