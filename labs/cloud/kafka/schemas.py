"""Event schemas for the reference ADT stream."""
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class AdtEvent(BaseModel):
    event_id: str = Field(min_length=8, max_length=64)
    patient_id: str = Field(min_length=4, max_length=64)
    event_type: Literal["admit", "discharge", "transfer"]
    event_time: datetime
    site_id: str = Field(min_length=2, max_length=32)
    ward: str | None = None
    schema_version: str = "1.0"
