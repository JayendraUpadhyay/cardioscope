"""Request and response contracts for the CardioScope API."""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class TabularInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    gender: int = Field(..., ge=1, le=2, description="1=Female, 2=Male")
    height: float = Field(..., ge=100, le=220, description="Height in cm")
    weight: float = Field(..., ge=20, le=300, description="Weight in kg")
    ap_hi: int = Field(..., ge=60, le=300, description="Systolic BP (mmHg)")
    ap_lo: int = Field(..., ge=30, le=200, description="Diastolic BP (mmHg)")
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def systolic_exceeds_diastolic(self):
        if self.ap_hi <= self.ap_lo:
            raise ValueError("Systolic BP must be greater than diastolic BP")
        return self


class ShapFactor(BaseModel):
    feature: str
    raw_value: Optional[float] = None
    direction: str
    magnitude: float


class TabularPrediction(BaseModel):
    risk_probability: float
    risk_label: str
    risk_percent: str
    top_factors: List[ShapFactor]
    bmi: float
    pulse_pressure: int
    disclaimer: str


class ECGInput(BaseModel):
    recording_index: int = Field(..., ge=0, le=186)


class ECGPrediction(BaseModel):
    recording_index: int
    anomaly_score: float
    risk_level: str
    interpretation: str
    percentile: float
    limitation_note: str
    waveform: List[float]
    waveform_sample_indices: List[int]
    segment_scores: List[float]
