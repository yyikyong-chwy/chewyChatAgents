from __future__ import annotations
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field

# ---------- Multi-slot planning & retrieval schemas ----------
class Budget(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    target: Optional[float] = None

class FilterSpec(BaseModel):
    # Hard filters applied before search; keep these extensible.
    species: Optional[Literal["dog", "cat"]] = None
    life_stage: Optional[Literal["puppy", "adult", "kitten", "senior"]] = None
    size: Optional[Literal["xs", "s", "m", "l", "xl"]] = None
    weight_lb: Optional[float] = None
    categories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    location_zip: Optional[str] = None
    availability_required: Optional[bool] = True

class CandidateProduct(BaseModel):
    sku: str
    title: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    # Scores from retrieval pipeline (for audit)
    score_vector: Optional[float] = None
    score_bm25: Optional[float] = None
    score_fused: Optional[float] = None
    score_rerank: Optional[float] = None
    score_rerank2: Optional[float] = None
    reasoning: Optional[str] = None

class SlotSpec(BaseModel):
    """One 'slot' in the bundle (e.g., core_food, leash, crate), with embedded candidates."""
    slot_id: str
    role: Literal["core", "single_purchase", "premium_addon", "optional"] = "optional"
    name: Optional[str] = None
    must_have: bool = True
    allow_substitution: bool = True

    # Enrichment fields
    product_type: Optional[str] = None                # e.g., "dry food", "leash", "sleeping mat"
    reason_for_suggestion: Optional[str] = None       # why this slot was suggested
    query_text: Optional[str] = None                  # context-sensitive query for embeddings
    soft_terms: Optional[List[str]] = None            # extra keywords to bias retrieval

    # Retrieval constraints
    filters: FilterSpec = Field(default_factory=FilterSpec)
    budget: Budget = Field(default_factory=Budget)

    # Retrieval results (potential candidates for this slot)
    candidates: List[CandidateProduct] = Field(default_factory=list)


class RetrievalParams(BaseModel):
    k_dense: int = 150
    k_bm25: int = 150
    fuse_k: int = 60              # RRF k-constant
    top_k: int = 12               # candidates kept per slot post-fusion
    apply_mmr: bool = True
    mmr_lambda: float = 0.7       # relevance vs. diversity
    rerank: Literal["none", "cross_encoder", "llm"] = "cross_encoder"
    rerank_top_k: int = 50

# --- New: bundle output schemas ---

class BundleItem(BaseModel):
    slot_id: str
    slot_name: Optional[str] = None
    sku: str
    title: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    quantity: int = 1
    # auditability
    score: Optional[float] = None          # the score used to select this item
    score_type: Literal["score_rerank", "score_fused", "score_vector", "none"] = "none"
    reasoning: Optional[str] = None        # carried over from CandidateProduct.reasoning

class RemovedSlot(BaseModel):
    slot_id: str
    slot_name: Optional[str] = None
    reason: str                              # e.g., "below threshold"
    best_candidate: Optional[Dict] = None    # shallow info for debugging (sku, title, score)


class ProposedBundle(BaseModel):
    items: List[BundleItem] = Field(default_factory=list)
    removed_slots: List[RemovedSlot] = Field(default_factory=list)
    subtotal: float = 0.0
    currency: Literal["USD"] = "USD"

# ---------- Aggregate state ----------

class bundleState(BaseModel):
    """Unified state for a pet parent's chat session/history.

    This object is intended to be persisted and reloaded to reconstruct
    context across sessions and to feed LangGraph nodes.
    """
    # Planner → (enricher fills queries) → retriever populates candidates
    slots: List[SlotSpec] = Field(default_factory=list)

    # (Optional) retrieval defaults; override per slot as needed
    retrieval_params: RetrievalParams = Field(default_factory=RetrievalParams)

    # New: final bundle result of bundle_creator_agent
    proposed_bundle: Optional[ProposedBundle] = None



