"""Request/response schemas for the inference API.

The request accepts only PRE_EVENT + SCENARIO feature codes (see
docs/prediction_time_contract.md). `extra="forbid"` means any unknown field —
including a banned outcome field like ``nkill`` — is rejected at the boundary.
All fields are optional; missing values are imputed by the pipeline, but more
scenario detail yields a more meaningful estimate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ScenarioRequest(BaseModel):
    """A fixed incident scenario plus a date range to sweep. `iyear`/`imonth` are
    NOT accepted here — they are the swept variables."""

    model_config = ConfigDict(extra="forbid")

    start_year: int = Field(..., ge=1970, le=2100)
    end_year: int = Field(..., ge=1970, le=2100)
    by: Literal["year", "month"] = "year"
    month: int | None = Field(None, ge=1, le=12, description="Month held fixed when by='year'")
    # Fixed scenario (same codes as /v1/predict, minus the date fields)
    country: int | None = None
    region: int | None = None
    specificity: int | None = None
    vicinity: int | None = None
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    attacktype1: int | None = None
    targtype1: int | None = None
    targsubtype1: int | None = None
    natlty1: int | None = None
    weaptype1: int | None = None
    weapsubtype1: int | None = None
    suicide: int | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def _check_range(self):
        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")
        return self


class ScenarioPoint(BaseModel):
    date: str = Field(..., description='"YYYY" (by=year) or "YYYY-MM" (by=month)')
    probability: float
    uncertainty_low: float
    uncertainty_high: float


class ScenarioResponse(BaseModel):
    model_version: str
    by: str
    scenario: str = Field(..., description="Plain-language restatement of the fixed scenario")
    points: list[ScenarioPoint]
    target: str
    disclaimer: str = Field(..., description="Non-forecast labelling — read this")


class Evidence(BaseModel):
    source: str = Field(..., description="Doc chunk id, e.g. model_card.md#evaluation")
    snippet: str
    score: float


class ExplainResponse(BaseModel):
    probability: float = Field(..., description="Calibrated P(>=1 fatality) — from Model A")
    uncertainty_low: float
    uncertainty_high: float
    model_version: str
    target: str
    scenario: str = Field(..., description="Plain-language restatement of the input")
    explanation: str = Field(..., description="Grounded narrative; never alters the number")
    evidence: list[Evidence] = Field(default_factory=list)
    generated_by: str = Field(..., description='"template" or the LLM id, e.g. anthropic:<model>')
    disclaimer: str
