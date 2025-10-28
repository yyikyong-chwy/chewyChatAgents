"""
bundle_creator_agent.py

Selects ONE best-fit product per slot (if any meet quality bar) and writes a
ProposedBundle onto the nppState. Slots whose best candidate is still a poor
match are excluded from the final bundle (with an explanation recorded).

Acceptance rule:
- Prefer CandidateProduct.score_rerank if present (0..1 from LLM evaluator).
- Otherwise fall back to score_fused, then score_vector.
- If best_score < MIN_ACCEPT, drop the slot from the bundle.

Override thresholds via environment variables:
    BUNDLE_MIN_ACCEPT_SCORE            (default 0.52)
    BUNDLE_MIN_ACCEPT_SCORE_CORE       (optional extra bar for core slots)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import os
from datetime import datetime, timezone

#custom imports
import LG_Agents.states.sessionState as sessionState
from LG_Agents.states.bundleState import SlotSpec, CandidateProduct, ProposedBundle, BundleItem, RemovedSlot
import LG_Agents.helperFunctions.agent_helper_function as agent_helper_function


def _pick_score(c: CandidateProduct) -> Tuple[Optional[float], str]:
    """Choose the best available score and its type label."""
    if c.score_rerank is not None:
        return float(c.score_rerank), "score_rerank"
    if c.score_fused is not None:
        return float(c.score_fused), "score_fused"
    if c.score_vector is not None:
        return float(c.score_vector), "score_vector"
    return None, "none"

def _best_candidate(cands: List[CandidateProduct]) -> Tuple[Optional[CandidateProduct], Optional[float], str]:
    """Return (cand, score, score_type) for the best available candidate list."""
    if not cands:
        return None, None, "none"
    # Sort by preferred score key
    def key(c: CandidateProduct):
        s, _ = _pick_score(c)
        # None scores sort last
        return (s is not None, s or -1.0)
    cc = sorted(list(cands), key=key, reverse=True)
    top = cc[0]
    s, t = _pick_score(top)
    return top, s, t

def _accept_threshold(slot: SlotSpec) -> float:
    base = float(os.getenv("BUNDLE_MIN_ACCEPT_SCORE", "0.52"))
    core_extra = os.getenv("BUNDLE_MIN_ACCEPT_SCORE_CORE")
    if core_extra is not None and slot.role == "core":
        try:
            return float(core_extra)
        except ValueError:
            pass
    return base

def bundle_creator_agent(state: sessionState) -> sessionState:
    """Build a ProposedBundle on state from per-slot candidates."""
    items: List[BundleItem] = []
    removed: List[RemovedSlot] = []

    for slot in getattr(state.bundle_state, "slots", []):
        if not isinstance(slot, SlotSpec):
            continue

        cand, score, score_type = _best_candidate(getattr(slot, "candidates", []) or [])
        thresh = _accept_threshold(slot)

        if cand is None or score is None or score < thresh:
            best_info = None if cand is None else {
                "sku": cand.sku,
                "title": cand.title,
                "score": score,
                "score_type": score_type,
            }
            removed.append(RemovedSlot(
                slot_id=slot.slot_id,
                slot_name=slot.name,
                reason=f"Excluded: best candidate below threshold ({score:.3f} < {thresh:.3f})" if score is not None else "Excluded: no viable candidates",
                best_candidate=best_info,
            ))
            continue

        items.append(BundleItem(
            slot_id=slot.slot_id,
            slot_name=slot.name,
            sku=cand.sku,
            title=cand.title,
            brand=cand.brand,
            category=cand.category,
            price=cand.price,
            score=score,
            score_type=score_type,
            reasoning=cand.reasoning,
            quantity=1,
        ))

    # Compute subtotal safely (ignore None prices)
    subtotal = sum(float(i.price) for i in items if i.price is not None)

    state.bundle_state.proposed_bundle = ProposedBundle(
        created_at=datetime.now(timezone.utc),
        items=items,
        removed_slots=removed,
        subtotal=subtotal,
        currency="USD",
    )
    return state


