# customerState.py

from __future__ import annotations
from typing import List, Optional, Literal, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------
# Customer Profile
# ---------------------------
class Customer(BaseModel):
    id: str = Field(..., description="Stable customer identifier")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    preferred_units: Literal["imperial", "metric"] = "imperial"
    timezone: Optional[str] = Field(None, description="IANA TZ, e.g. America/Los_Angeles")
    zip_code: Optional[str] = None

    geo: GeoLocation = None
    weather_now: Optional[WeatherObservation] = None
    #weather_forecast: Optional[WeatherForecast] = None


# ---------------------------
# Location
# ---------------------------
class GeoLocation(BaseModel):
    city: Optional[str] = None
    region: Optional[str] = Field(None, description="State/Province code, e.g. CA")
    country: str = "US"
    postal_code: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    timezone: Optional[str] = None  # keep here too (helpful if weather provider returns TZ)

# ---------------------------
# Weather (current + forecast)
# ---------------------------
class WeatherObservation(BaseModel):
    provider: Optional[str] = Field(None, description="e.g., open-meteo")
    observed_at: Optional[datetime] = None
    weather_code: Optional[int] = Field(None, description="Provider-specific or WMO code")
    description: Optional[str] = None

    temp_c: Optional[float] = None
    temp_f: Optional[float] = None

    windspeed_kph: Optional[float] = None
    windspeed_mph: Optional[float] = None

    humidity_pct: Optional[int] = Field(None, ge=0, le=100)
    precip_prob_pct: Optional[int] = Field(None, ge=0, le=100)
    uv_index: Optional[float] = Field(None, ge=0)

    # Where this weather applies
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    timezone: Optional[str] = None

    @model_validator(mode="after")
    def _derive_units(self) -> "WeatherObservation":
        if self.temp_c is not None and self.temp_f is None:
            self.temp_f = round(self.temp_c * 9 / 5 + 32, 1)
        if self.temp_f is not None and self.temp_c is None:
            self.temp_c = round((self.temp_f - 32) * 5 / 9, 1)

        if self.windspeed_kph is not None and self.windspeed_mph is None:
            self.windspeed_mph = round(self.windspeed_kph * 0.621371, 1)
        if self.windspeed_mph is not None and self.windspeed_kph is None:
            self.windspeed_kph = round(self.windspeed_mph / 0.621371, 1)
        return self







