# policy_models.py
from pydantic import BaseModel, Field
from typing import List

class BundlePricingGuidelines(BaseModel):
    basic_target_min: float = 30.0
    basic_target_max: float = 80.0
    premium_target_min: float = 120.0
    premium_target_max: float = 260.0

class BundleRules(BaseModel):
    core_items: List[str]
    single_purchase_items: List[str] = []
    premium_addons: List[str] = []
    pricing: BundlePricingGuidelines = Field(default_factory=BundlePricingGuidelines)

class PolicyRef(BaseModel):
    ruleset_id: str
    version: str
    hash: str
