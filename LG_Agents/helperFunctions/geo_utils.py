# geo_utils.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
import re
from dataclasses import dataclass
import math
import datetime as dt

# Optional fallbacks
import httpx

# pip install pgeocode timezonefinder httpx
import pgeocode
from timezonefinder import TimezoneFinder

_ZIP5 = re.compile(r"\b(\d{5})\b")
_tf = TimezoneFinder()

@lru_cache(maxsize=5000)
def zip_to_latlon(zip_or_plus4: str, country: str = "US") -> Optional[Dict[str, Any]]:
    """
    Returns:
      {
        'lat': float,
        'lon': float,
        'city': str|None,
        'state': str|None,
        'tz': str|None,
        'source': 'pgeocode'|'zippopotam'
      }
    or None if not found.
    """
    if not zip_or_plus4:
        return None

    # normalize to 5-digit ZIP for US
    m = _ZIP5.search(str(zip_or_plus4))
    if not m:
        return None
    zip5 = m.group(1)

    # 1) Offline: pgeocode (centroid of the ZIP code area)
    try:
        nomi = pgeocode.Nominatim(country.lower())
        row = nomi.query_postal_code(zip5)
        if row is not None and row.latitude == row.latitude and row.longitude == row.longitude:
            lat = float(row.latitude)
            lon = float(row.longitude)
            tz = _tf.timezone_at(lat=lat, lng=lon)
            return {
                "lat": lat,
                "lon": lon,
                "city": (row.place_name if isinstance(row.place_name, str) else None),
                "state": (row.state_code if isinstance(row.state_code, str) else None),
                "tz": tz,
                "source": "pgeocode",
            }
    except Exception:
        pass

    # 2) Fallback: free API (no key). Note: no SLA; use paid geocoding in prod if you need guarantees.
    try:
        url = f"https://api.zippopotam.us/us/{zip5}"
        with httpx.Client(timeout=6.0) as client:
            r = client.get(url)
            if r.status_code == 200:
                data = r.json()
                place = (data.get("places") or [None])[0] or {}
                lat = float(place.get("latitude"))
                lon = float(place.get("longitude"))
                tz = _tf.timezone_at(lat=lat, lng=lon)
                return {
                    "lat": lat,
                    "lon": lon,
                    "city": place.get("place name"),
                    "state": place.get("state abbreviation"),
                    "tz": tz,
                    "source": "zippopotam",
                }
    except Exception:
        pass

    return None


# -----------------------------
# 1) Weather "tool"
#    - Uses Open-Meteo (no API key)
# -----------------------------
@dataclass
class WeatherResult:
    temperature_c: float
    temperature_f: float
    windspeed_kmh: float
    windspeed_mph: float
    weather_code: int
    precipitation_prob: Optional[int]
    city: Optional[str]
    tz: str
    observed_at: str  # ISO timestamp

class WeatherTool:
    """Simple tool you can also expose to an LLM later if you want."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    @staticmethod
    def get_current_from_zip(zip_code: str) -> WeatherResult:
        latlon = zip_to_latlon(zip_code)
        return WeatherTool.get_current(lat=latlon["lat"], lon=latlon["lon"], tz=latlon["tz"], city=latlon["city"])

    @staticmethod    
    def get_current(*, lat: float, lon: float, tz: str = "UTC", city: Optional[str] = None) -> WeatherResult:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": "precipitation_probability",
            "timezone": tz or "UTC",
        }
        with httpx.Client(timeout=8.0) as client:
            r = client.get(WeatherTool.BASE_URL, params=params)
            r.raise_for_status()
            data = r.json()

        cur = data.get("current_weather", {}) or {}
        hourly = data.get("hourly", {}) or {}

        # Map precip prob at the nearest hour to "current" time index (best-effort)
        precip_prob = None
        try:
            times = hourly.get("time", [])
            probs = hourly.get("precipitation_probability", [])
            now_iso = cur.get("time") or dt.datetime.now(dt.timezone.utc).isoformat()
            # simple nearest match
            if times and probs:
                # find exact index if present
                if now_iso in times:
                    i = times.index(now_iso)
                else:
                    # nearest by absolute delta
                    i = min(range(len(times)), key=lambda k: abs(dt.datetime.fromisoformat(times[k]) - dt.datetime.fromisoformat(now_iso)))
                precip_prob = int(probs[i]) if probs[i] is not None else None
        except Exception:
            precip_prob = None

        temp_c = float(cur.get("temperature", math.nan))
        temp_f = temp_c * 9 / 5 + 32 if not math.isnan(temp_c) else math.nan
        wind_kmh = float(cur.get("windspeed", 0.0))
        wind_mph = wind_kmh * 0.621371

        return WeatherResult(
            temperature_c=temp_c,
            temperature_f=round(temp_f, 1) if not math.isnan(temp_f) else temp_f,
            windspeed_kmh=round(wind_kmh, 1),
            windspeed_mph=round(wind_mph, 1),
            weather_code=int(cur.get("weathercode", 0)),
            precipitation_prob=precip_prob,
            city=city,
            tz=params["timezone"],
            observed_at=cur.get("time") or dt.datetime.now(dt.timezone.utc).isoformat(),
        )



#Testing
if __name__ == "__main__":
    print(zip_to_latlon("98075")) #Sammamish, WA
    print(zip_to_latlon("33503")) #Port Charlotte, FL
    print(zip_to_latlon("78701")) #Austin, TX
    print(zip_to_latlon("55418")) #Minneapolis, MN

    latloncity = zip_to_latlon("55418")

    w = WeatherTool.get_current(lat=float(latloncity["lat"]), lon=float(latloncity["lon"]), tz=latloncity["tz"], city=latloncity["city"])
    print(w)

