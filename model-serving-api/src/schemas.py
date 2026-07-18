"""Pydantic request/response models for the risk-prediction API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    pregnancies: int = Field(ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(
        ge=0, le=300, description="Plasma glucose concentration (mg/dL); 0 = not recorded")
    blood_pressure: float = Field(
        ge=0, le=200, description="Diastolic blood pressure (mm Hg); 0 = not recorded")
    skin_thickness: float = Field(
        ge=0, le=100, description="Triceps skinfold thickness (mm); 0 = not recorded")
    insulin: float = Field(
        ge=0, le=900, description="2-hour serum insulin (mu U/mL); 0 = not recorded")
    bmi: float = Field(ge=0, le=70, description="Body mass index; 0 = not recorded")
    diabetes_pedigree_function: float = Field(
        ge=0, le=3, description="Genetic risk score based on family history")
    age: int = Field(ge=1, le=120, description="Age in years")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "pregnancies": 2, "glucose": 130, "blood_pressure": 70,
            "skin_thickness": 25, "insulin": 90, "bmi": 32.5,
            "diabetes_pedigree_function": 0.45, "age": 35,
        }
    })


class PredictionResponse(BaseModel):
    risk_probability: float = Field(description="P(diabetes) from the model, in [0, 1]")
    risk_label: str = Field(description='"high" if risk_probability >= threshold else "low"')
    threshold: float = Field(description="Decision threshold applied to risk_probability")
    model_holdout_auc: float = Field(
        description="AUC of this model on a held-out test split, for context")
