from pydantic import BaseModel, Field
from typing_extensions import Annotated
import typing as t

# -----------------------------
# Configuration (defaults)
# -----------------------------
DEFAULT_SEED = "https://www.chewy.com/vet-care"
ALLOWED_DOMAIN = "chewy.com"
DEFAULT_MODEL = "gpt-4o-mini"


# -----------------------------
# Pydantic State (extensible)
# -----------------------------
class ConfigModel(BaseModel):
    seed: str = Field(default=DEFAULT_SEED)
    allowed_domain: str = Field(default=ALLOWED_DOMAIN)
    max_pages: int = Field(default=10, ge=1, description="Max number of pages to fetch in this short crawl")
    only_vetcare_paths: bool = Field(default=True, description="If true, only crawl URLs containing /vet-care/")
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.0)


class LocationRecord(BaseModel):
    url: str
    name_guess: t.Optional[str] = None
    source: t.Optional[str] = None


class LocationDiscoveryState(BaseModel):
    """Graph state; fields are annotated with reducers so LangGraph can merge node outputs.
    You can safely extend this model later (e.g., add `location_details`, `tokens_used`, etc.).
    """
    # crawl queues
    queue: Annotated[list[str], add] = Field(default_factory=list)
    visited: Annotated[set[str], or_merge] = Field(default_factory=set)

    # current page context
    current_url: t.Optional[str] = None
    html: t.Optional[str] = None
    anchors: Annotated[list[dict], add] = Field(default_factory=list)  # {href, text}

    # extraction products
    candidates: Annotated[list[str], add] = Field(default_factory=list)
    results: Annotated[list[LocationRecord], add] = Field(default_factory=list)

    # tracing / notes
    llm_notes: Annotated[list[str], add] = Field(default_factory=list)
    errors: Annotated[list[str], add] = Field(default_factory=list)

    # configuration
    config: ConfigModel = Field(default_factory=ConfigModel)


# -----------------------------
# LLM output schema (structured)
# -----------------------------
class LinkItem(BaseModel):
    url: str
    is_location: bool = False
    name_guess: t.Optional[str] = None
    reason: t.Optional[str] = None


class LinkMiningOutput(BaseModel):
    location_links: list[LinkItem] = Field(default_factory=list)
    crawl_next: list[str] = Field(default_factory=list)
    notes: t.Optional[str] = None
