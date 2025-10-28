# pet_insurance_agent.py
from __future__ import annotations

import os, json, math, re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import AIMessage

# --- State wiring (both package-style and local fallbacks)
from LG_Agents.states.sessionState import SessionStateModel
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState
from LG_Agents.chatAgents.chat_helper_function import _augment_query_with_history
from LG_Agents.helperFunctions.embedding_retrieval_function import retrieve, Chunk, grounded_score_from_evidence

# OpenAI for answer synthesis + query embeddings
OPENAI_MODEL_ANS = os.getenv("CJ_INSURANCE_ANSWER_MODEL", "gpt-4o-mini")
OPENAI_MODEL_EMB = os.getenv("CJ_INSURANCE_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TOP_K = int(os.getenv("CJ_INSURANCE_TOP_K", "6"))
MAX_TOK_CONTEXT = int(os.getenv("CJ_INSURANCE_MAX_CONTEXT", "1800"))  # soft guard


INSURANCE_EMBEDDINGS_INDEX_PATH = os.getenv("CJ_INSURANCE_INDEX", "C:/genAIProjects/new-pet-parent-poc/backend/LG_Agents/embeddings/pet_insurance_index.jsonl")
USE_VERBATIM_QUOTES = False  # keep paraphrased to avoid long quotes


# ------------------------
# Answer synthesis
# ------------------------
_SYSTEM = """You are Chewy's Pet Insurance assistant.
Answer ONLY using the supplied CONTEXT. If a user asks for legal or policy details not in context, say you don't have that info and suggest contacting Chewy Insurance support.
Keep replies concise (4–8 sentences). Use plain language, no legal advice.
When relevant, include limits, waiting periods, exclusions, claim windows, network rules, and state variations if present in CONTEXT.
Cite the section names you used like [Section: Claims > Filing a claim]. Do not invent citations."""

def _build_context(chunks: List[Chunk]) -> str:
    if not chunks:
        return ""
    ctx = []
    n_chars = 0
    for ch in chunks:
        add = f"[{ch.section}]\n{ch.text}\n"
        if n_chars + len(add) > MAX_TOK_CONTEXT * 5:  # coarse char cap
            break
        ctx.append(add)
        n_chars += len(add)
    return "\n".join(ctx).strip()

def _llm_answer(question: str, chunks: List[Chunk]) -> str:
    context = _build_context(chunks)
    if not context:
        return ("I couldn’t find insurance details in my reference file yet. "
                "Try rephrasing your question or ask me about coverage, claims, pricing, waiting periods, or what’s excluded.")
    if not OPENAI_API_KEY:
        # Deterministic, short fallback: stitch together the top chunks
        parts = []
        used = []
        for ch in chunks[:3]:
            parts.append(ch.text.strip())
            used.append(ch.section)
        answer = " ".join(parts[:2])
        if not answer:
            answer = "I found a few relevant sections, but I need more specific details."
        answer += " " + " ".join([f"[Section: {s}]" for s in used[:2]])
        return answer

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL_ANS,
        messages=messages,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

# ------------------------
# MAIN NODE
# ------------------------
def pet_insurance_agent(state: SessionStateModel) -> SessionStateModel:
    cjs: ChewyJourneyChatState = state.chewy_journey_chat_state
    user_q = (cjs.last_user_message or "").strip()

    # Gentle red lines
    if re.search(r"\b(emergency|bleeding|not breathing|collapse(d)?)\b", user_q, flags=re.I):
        cjs.agent_response = ("If your pet may be in danger, please contact an emergency veterinarian immediately. "
                              "For insurance questions (coverage, claims, waiting periods), I can help here.")
        cjs.route = "insurance"; cjs.domain = "insurance"
        state.chewy_journey_chat_state = cjs
        return state

    hybrid_query = _augment_query_with_history(user_q, cjs, max_chars=1000)

    # Retrieve → answer
    chunks = retrieve(hybrid_query, k=TOP_K, embeddings_index_path=INSURANCE_EMBEDDINGS_INDEX_PATH)
    answer = _llm_answer(user_q, chunks)

    # Add a standard coverage disclaimer if we referenced insurance
    if re.search(r"\b(coverage|claim|claims|deductible|reimbursement|waiting period|exclusion|premium|pricing)\b", user_q, re.I):
        answer += "\n\nNote: Availability and terms may vary by state and policy. For billing or policy-specific details, I can connect you to Chewy Insurance support."

    #determining if the agent response is grounded, complete, and safe
    used_vectors = bool(OPENAI_API_KEY) and any(ch.vector for ch in chunks)
    grounded = grounded_score_from_evidence(answer, chunks, used_vectors)


    return {
        "messages": [AIMessage(content=answer)],
        "chewy_journey_chat_state": {
            "agent_response": answer,            
            "route": "insurance",
            "confidence": 1.0,
            "grounded_score": grounded,
            "debug": {
                "top_chunk_sections": [ch.section for ch in chunks],
                "used_vectors": bool(OPENAI_API_KEY) and any(ch.vector for ch in chunks),
                "index_path": INSURANCE_EMBEDDINGS_INDEX_PATH,
                "model_answer": OPENAI_MODEL_ANS,
                "model_embed": OPENAI_MODEL_EMB,
            },
        }
    }

# -------------- Local smoke test
if __name__ == "__main__":
    
    from pathlib import Path
    from LG_Agents.states.sessionState import load_state

    # Minimal smoke test without real state loader; replace with your state loader as needed.
    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    state = load_state(json_path)
    last_user_message="How is pet insurance different from human health insurance?"
    #last_user_message="Is pet insurance worth it?"
    state.chewy_journey_chat_state=ChewyJourneyChatState(last_user_message=last_user_message)

    state = pet_insurance_agent(state)
    print("\n--- AGENT REPLY ---")
    print(state.chewy_journey_chat_state.agent_response)
    print("\nDEBUG:", state.chewy_journey_chat_state.debug.get("insurance"))
