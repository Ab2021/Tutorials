"""Train a tiny logistic model on synthetic data and write app/model.json.

This is a teaching example, not a clinical model. It keeps the stack runnable
without scikit-learn and makes the math inspectable.
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path

FEATURES = ["age", "systolic_bp", "creatinine", "prior_admissions"]
random.seed(7)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def make_data(n: int = 2000):
    rows = []
    for _ in range(n):
        age = random.gauss(65, 12)
        sbp = random.gauss(135, 18)
        creat = max(0.3, random.gauss(1.1, 0.4))
        prior = max(0, int(random.gauss(0.6, 0.8)))
        z = -6 + 0.04 * age + 0.01 * sbp + 0.3 * creat + 0.4 * prior
        y = 1 if random.random() < sigmoid(z) else 0
        rows.append(([age, sbp, creat, prior], y))
    return rows


def train(rows, epochs: int = 300, lr: float = 0.01):
    w = [0.0] * len(FEATURES)
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * len(FEATURES)
        gb = 0.0
        for x, y in rows:
            p = sigmoid(b + sum(wi * xi for wi, xi in zip(w, x)))
            err = p - y
            for i in range(len(w)):
                gw[i] += err * x[i]
            gb += err
        n = len(rows)
        w = [wi - lr * gi / n for wi, gi in zip(w, gw)]
        b -= lr * gb / n
    return w, b


if __name__ == "__main__":
    rows = make_data()
    w, b = train(rows)
    payload = {
        "version": "0.1.0",
        "intercept": b,
        "coefficients": dict(zip(FEATURES, w)),
        "threshold": 0.2,
    }
    Path("app/model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
