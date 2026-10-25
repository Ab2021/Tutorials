from __future__ import annotations
from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(ge=0, le=120)
    systolic_bp: float = Field(ge=40, le=300)
    creatinine: float = Field(ge=0.1, le=20)
    prior_admissions: float = Field(ge=0, le=100)


class Prediction(BaseModel):
    risk: float
    threshold: float
    decision: str
    model_version: str
