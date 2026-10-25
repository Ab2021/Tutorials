"""Deterministic logistic risk model used by the reference service."""
from __future__ import annotations
import json
import math
from pathlib import Path


class RiskModel:
    def __init__(self, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.version: str = payload["version"]
        self.intercept: float = float(payload["intercept"])
        self.coefficients: dict[str, float] = {k: float(v) for k, v in payload["coefficients"].items()}
        self.threshold: float = float(payload["threshold"])

    def predict(self, features: dict[str, float]) -> float:
        z = self.intercept + sum(self.coefficients[k] * float(features[k]) for k in self.coefficients)
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
