from typing import List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState



def _ensure_messages_list(cjs: ChewyJourneyChatState) -> List[BaseMessage]:
    msgs = getattr(cjs, "messages", None)
    if msgs is None:
        msgs = []
    elif not isinstance(msgs, list):
        msgs = list(msgs)
    # Normalize: ensure only BaseMessage-like or str are included; coerce dicts to strings
    norm: List[BaseMessage] = []
    for m in msgs:
        if isinstance(m, BaseMessage):
            norm.append(m)
        elif isinstance(m, dict) and "content" in m and "type" in m:
            # crude conversion if using LangGraph message dicts
            role = m.get("type")
            content = m.get("content", "")
            if role == "human":
                norm.append(HumanMessage(content=content))
            elif role == "ai":
                norm.append(AIMessage(content=content))
        elif isinstance(m, str):
            # fallback
            norm.append(HumanMessage(content=m))
    return norm

def _recent_history_text(cjs: ChewyJourneyChatState, max_chars: int = 1400, max_turns: int = 6) -> str:
    """Flatten last few turns into a compact text block."""
    msgs = _ensure_messages_list(cjs)
    if not msgs:
        return ""
    # take from the end, collect up to max_turns exchanges
    buf: List[str] = []
    turns = 0
    for m in reversed(msgs):
        prefix = "Assistant" if isinstance(m, AIMessage) else ("User" if isinstance(m, HumanMessage) else "Msg")
        buf.append(f"{prefix}: {m.content}")
        # crude turn counting: count user lines as turns
        if isinstance(m, HumanMessage):
            turns += 1
            if turns >= max_turns:
                break
    text = "\n".join(reversed(buf))
    return text[-max_chars:] if len(text) > max_chars else text

def _augment_query_with_history(user_q: str, cjs: ChewyJourneyChatState, max_chars: int = 800) -> str:
    """Hybrid query: current question + last user lines for retrieval."""    
    # Extract recent user-side questions for keyword boost
    msgs = _ensure_messages_list(cjs)
    recent_user_bits = [m.content for m in msgs if isinstance(m, HumanMessage)]
    recent_tail = " ".join(recent_user_bits[-3:]) if recent_user_bits else ""
    combo = " ".join([user_q, recent_tail]).strip()
    if len(combo) > max_chars:
        combo = combo[:max_chars]
    return combo