"""Request/response schemas for the inference API.

The request accepts only PRE_EVENT + SCENARIO feature codes (see
docs/prediction_time_contract.md). `extra="forbid"` means any unknown field —
including a banned outcome field like ``nkill`` — is rejected at the boundary.
All fields are optional; missing values are imputed by the pipeline, but more
scenario detail yields a more meaningful estimate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # PRE_EVENT
    iyear: int | None = Field(None, description="Incident year")
    imonth: int | None = Field(None, ge=1, le=12)
    country: int | None = Field(None, description="GTD country code")
    region: int | None = Field(None, description="GTD region code")
    specificity: int | None = None
    vicinity: int | None = None
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    # SCENARIO
    attacktype1: int | None = None
    targtype1: int | None = None
    targsubtype1: int | None = None
    natlty1: int | None = Field(None, description="GTD nationality code of target")
    weaptype1: int | None = None
    weapsubtype1: int | None = None
    suicide: int | None = Field(None, ge=0, le=1)


class PredictResponse(BaseModel):
    probability: float = Field(..., description="Calibrated P(>=1 fatality)")
    uncertainty_low: float
    uncertainty_high: float
    model_version: str
    target: str
    disclaimer: str
