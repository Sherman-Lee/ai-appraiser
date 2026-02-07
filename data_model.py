from enum import Enum
from pydantic import BaseModel, Field
from env_config import DEFAULT_CURRENCY


# --- Data Models ---
class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"


class ValuationResponse(BaseModel):
    estimated_value: float
    currency: Currency = Field(
        Currency(DEFAULT_CURRENCY),
        description="Currency code (ISO 4217)",
    )
    reasoning: str
    search_urls: list[str]


class ValuationPerImage(BaseModel):
    image_index: int
    valuation: ValuationResponse


class MultiValuationResponse(BaseModel):
    results: list[ValuationPerImage]