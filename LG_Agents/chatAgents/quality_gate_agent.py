# cj_agent_orchestrator.py (new imports)
import json
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel

#custom imports
from LG_Agents.states.sessionState import SessionStateModel
from langchain_core.messages import AIMessage
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState

class GateReview(BaseModel):
    grounded: float
    complete: float
    safe: float
    escalate: bool
    reason: str

HALLUC_PATTERNS = {
  "ai_disclaimer": [
    r"\bas an ai\b", r"\bas an ai language model\b",
    r"\bi (cannot|can't) (access|browse)\b", r"\bknowledge cutoff\b"
  ],
  "speculative_then_claim": [
    r"\b(probably|likely|i (think|believe)|should|might)\b.*\b(\d{2,}|guarantee|exact|always|never)\b"
  ],
  "fabricated_ids": [
    r"\b(order|tracking|ticket|sku|policy)\s*(id|#|number)?\s*[:\-]?\s*[A-Z0-9]{6,}\b",
    r"\b[A-Z]{2,}\d{6,}\b"  # generic all-caps alphanumeric
  ],
  "fake_citation": [
    r"\[\d{1,2}\]", r"\([A-Z][a-zA-Z]+,\s*\d{4}\)",
    r"\baccording to (wikipedia|fda|who|ups|fedex|amazon)\b(?!.*https?://)"
  ],
  "authority_overreach": [
    r"\bi (issued|processed|applied|updated|changed) (a |the )?(refund|credit|order|discount)\b",
    r"\bi (confirmed|verified) with (the )?(vet|pharmacy)\b"
  ],
  "contradiction": [
    r"\bi (cannot|can't|don'?t have) access\b.*\b(but|however)\b.*\b(updated|see|found|checked)\b"
  ],
  "overprecise_contact": [
    r"\b1[-\s]800[-\s]\d{3}[-\s]\d{4}\b", r"\b\d{3}[-\s]\d{3}[-\s]\d{4}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
  ],
  "medical_risk": [
    r"\b\d+(\.\d+)?\s*(mg|ml|μg|mcg)\/kg\b", r"\b(diagnos(e|is|ed)|prescrib(e|ed))\b"
  ],
  "policy_absolute": [
    r"\b(company|chewy)\s+policy\b", r"\b(100%|always|never)\b.*\b(guarantee|refund|delivery)\b"
  ],
}

import re
# this function will be used to score the hallucination of the agent response
def halluc_score(text: str) -> tuple[float, dict]:
    hits = {}
    score = 0.0
    for k, pats in HALLUC_PATTERNS.items():
        for p in pats:
            if re.search(p, text, flags=re.I):
                hits.setdefault(k, []).append(p)
    # Weight sensitive categories higher
    weights = {
        "authority_overreach": 2.0, "medical_risk": 2.0, "contradiction": 1.5,
        "fabricated_ids": 1.2, "fake_citation": 1.0, "ai_disclaimer": 0.8,
        "policy_absolute": 0.8, "speculative_then_claim": 0.6, "overprecise_contact": 0.6
    }
    for k, ps in hits.items():
        score += weights.get(k, 0.5) * len(ps)
    # Normalize to 0..1 rough score
    score = min(1.0, score / 4.0)
    return score, hits


# this agent will be used to gate the quality of the agent responses
# it will be used to determine if the agent response is grounded, complete, and safe
# it will also be used to determine if the agent response should be escalated to a human
# it will be used to determine if the agent response is hallucinated, or if it is even addressing the user's question


# ---------- NEW: quality gate ----------
def quality_gate_agent(state: SessionStateModel) -> SessionStateModel:
    cj = state.chewy_journey_chat_state
    user_q = (cj.last_user_message or "").strip()
    # Pull the last AI message content if present
    last_ai = next((m.content for m in reversed(state.messages) if isinstance(m, AIMessage)), "")
    route = (cj.route or "").lower()

    # 1) Compute hallucination score + pattern hits (0..1, higher = riskier)
    hscore, hhits = halluc_score(last_ai)  # ← now leveraged

    # 2) Quick heuristics
    risky_route = route in {"clinic", "pharmacy", "insurance"}        # raise the bar here
    too_short = len(last_ai.split()) < 4                 # likely non-answer

    # Optional: upstream retrieval signal (0..1)
    grounded_score = getattr(cj, "grounded_score", None)

    # 3) Fast-pass if clearly OK:
    #    - not in a risky route
    #    - decent length
    #    - low hallucination risk
    #    - grounded enough (when available)
    if (not risky_route
        and not too_short
        and hscore <= 0.2
        and (grounded_score is None or grounded_score >= 0.6)):
        cj.escalate = False

        return state

    # 4) Otherwise ask a small judge to rate it (LLM-as-a-judge). If it fails, be conservative.
    try:
        from openai import OpenAI
        client = OpenAI()
        rubric = f"""
        You are a strict reviewer for customer support replies.
        USER: {user_q}
        ASSISTANT: {last_ai}

        Score 0..1:
        - grounded: is the answer factual/consistent with likely policy or retrieved facts?
        - complete: does it directly answer the question (not generic fluff)?
        - safe: any medical/legal/compliance risks for a pet-care retailer? (1=safe)

        Return JSON with fields:
        grounded, complete, safe, escalate (bool), reason (short).
        Escalate if grounded<0.6 OR complete<0.6 OR safe<0.9; for clinic/pharmacy, escalate if any <0.8.
        """
        resp = client.responses.parse(
            model="gpt-4o-mini",
            input=rubric,
            text_format=GateReview,  # JSON mode for Responses API
            )
        result: GateReview = resp.output_parsed
        j = result.model_dump()
    except Exception as e:
        j = {
            "grounded": grounded_score if grounded_score is not None else 0.5,
            "complete": 0.5,
            "safe": 0.7,
            "escalate": risky_route,  # conservative on risky routes
            "reason": f"judge_error:{e}",
        }

    # 5) Blend in hallucination risk to final decision
    #    - escalate if hallucination risk is medium/high
    #    - make thresholds stricter on sensitive routes
    halluc_escalate = (hscore >= 0.5) or (risky_route and hscore >= 0.35)

    if risky_route:
        j["escalate"] = (
            j.get("escalate", False)
            or halluc_escalate
            or j.get("grounded", 0.0) < 0.8
            or j.get("complete", 0.0) < 0.8
            or j.get("safe", 0.0) < 0.95
        )
    else:
        j["escalate"] = (
            j.get("escalate", False)
            or halluc_escalate
            or j.get("grounded", 0.0) < 0.6
            or j.get("complete", 0.0) < 0.6
            or j.get("safe", 0.0) < 0.9
        )

    return {
        "chewy_journey_chat_state": {
            "escalate": bool(j.get("escalate")),
        }
    }


# ---------- NEW: human handoff ----------
def human_handoff(state: SessionStateModel) -> SessionStateModel:
    # Signal your UI to hand off; include a friendly customer-facing message.
    msg = "I’m looping in a specialist to make sure you get the most accurate help."
    state.messages.append(AIMessage(content=msg))
    # Option A: raise an interrupt your UI already handles
    raise GraphInterrupt({"type": "HUMAN_HANDOFF", "reason": getattr(state.chewy_journey_chat_state, "quality_debug", {})})
    # Option B (if you prefer not to interrupt): set a flag and END; your UI polls it.
    # state.chewy_journey_chat_state.handoff_requested = True
    # return state
