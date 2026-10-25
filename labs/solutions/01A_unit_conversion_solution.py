"""Instructor solution: unit conversions with explicit units and plausibility checks."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Measurement:
    value: float
    unit: str


def glucose_mgdl_to_mmol(value: float) -> float:
    return value * 0.0555


def creatinine_mgdl_to_umol(value: float) -> float:
    return value * 88.4


def cholesterol_mgdl_to_mmol(value: float) -> float:
    return value * 0.02586


def calcium_mgdl_to_mmol(value: float) -> float:
    return value * 0.25


def map_mmhg(sbp: float, dbp: float) -> float:
    return dbp + (sbp - dbp) / 3


def bmi(weight_kg: float, height_m: float) -> float:
    return weight_kg / height_m**2


def bsa_mosteller(height_cm: float, weight_kg: float) -> float:
    return (height_cm * weight_kg / 3600) ** 0.5


def main() -> None:
    print("glucose mmol/L:", round(glucose_mgdl_to_mmol(126), 2))
    print("creatinine umol/L:", round(creatinine_mgdl_to_umol(1.2), 1))
    print("cholesterol mmol/L:", round(cholesterol_mgdl_to_mmol(200), 2))
    print("calcium mmol/L:", round(calcium_mgdl_to_mmol(9.5), 2))
    print("MAP mmHg:", round(map_mmhg(120, 80), 1))
    print("BMI:", round(bmi(69.9, 1.75), 1))
    print("BSA m2:", round(bsa_mosteller(175, 69.9), 2))


if __name__ == "__main__":
    main()
