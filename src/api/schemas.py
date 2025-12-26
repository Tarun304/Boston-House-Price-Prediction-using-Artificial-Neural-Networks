"""Pydantic models for request/response validation"""

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """Input features for house price prediction"""

    AGE: float = Field(
        ..., description="Proportion of owner-occupied units built prior to 1940"
    )
    B: float = Field(..., description="1000(Bk - 0.63)^2")
    CHAS: int = Field(..., description="Charles River dummy variable (1=Yes, 0=No)")
    CRIM: float = Field(..., description="Per capita crime rate by town")
    DIS: float = Field(
        ..., description="Weighted distances to five Boston employment centres"
    )
    INDUS: float = Field(
        ..., description="Proportion of non-retail business acres per town"
    )
    LSTAT: float = Field(..., description="% lower status of the population")
    NOX: float = Field(
        ..., description="Nitric oxides concentration (parts per 10 million)"
    )
    PTRATIO: float = Field(..., description="Pupil-teacher ratio by town")
    RAD: int = Field(..., description="Index of accessibility to radial highways")
    RM: float = Field(..., description="Average number of rooms per dwelling")
    TAX: float = Field(..., description="Full-value property-tax rate per $10,000")
    ZN: float = Field(
        ...,
        description="Proportion of residential land zoned for lots over 25,000 sq.ft.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "AGE": 65.2,
                "B": 396.90,
                "CHAS": 0,
                "CRIM": 0.00632,
                "DIS": 4.0900,
                "INDUS": 2.31,
                "LSTAT": 4.98,
                "NOX": 0.538,
                "PTRATIO": 15.3,
                "RAD": 1,
                "RM": 6.575,
                "TAX": 296.0,
                "ZN": 18.0,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predicted_price: float = Field(..., description="Predicted house price in USD")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="API status: healthy or unhealthy")
    model_loaded: bool = Field(
        ..., description="Whether model is ready to serve predictions"
    )
