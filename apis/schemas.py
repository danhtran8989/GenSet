from pydantic import BaseModel, Field
from typing import List, Optional


class GenerateDatasetRequest(BaseModel):
    platform: str = Field("ollama", example="ollama")
    language: str = Field("vietnamese", example="vietnamese")
    domain: str = Field("medical assistant conversation", example="medical assistant conversation")
    labels: List[str] = Field(
        default=[
            "guide_symptoms", "guide_department", "guide_first_aid", "get_doctor",
            "get_patient_admin", "get_appointment", "explain_procedure", "guide_insurance",
            "guide_navigation", "explain_medical_education", "explain_document", "guide_booking"
        ],
        example=[
            "guide_symptoms", "guide_department", "guide_first_aid", "get_doctor",
            "get_patient_admin", "get_appointment", "explain_procedure", "guide_insurance",
            "guide_navigation", "explain_medical_education", "explain_document", "guide_booking"
        ],
        description="List of 12 medical intent classes"
    )
    num_samples: int = Field(200, ge=10, le=5000, example=300)
    model: Optional[str] = Field(None, example="gemma4:26b")
    temperature: float = Field(0.7, ge=0.0, le=1.0, example=0.7)
    delay: float = Field(0.8, ge=0.0, example=0.6)
    multilingual: bool = Field(True, example=True)
    balance_labels: bool = Field(True, example=True)

    class Config:
        json_schema_extra = {
            "example": {
                "platform": "ollama",
                "language": "vietnamese",
                "domain": "medical assistant conversation",
                "labels": [
                    "guide_symptoms", "guide_department", "guide_first_aid", "get_doctor",
                    "get_patient_admin", "get_appointment", "explain_procedure", "guide_insurance",
                    "guide_navigation", "explain_medical_education", "explain_document", "guide_booking"
                ],
                "num_samples": 300,
                "model": "gemma4:26b",
                "temperature": 0.7,
                "multilingual": True,
                "balance_labels": True
            }
        }