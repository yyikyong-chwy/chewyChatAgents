# reducers.py
from typing import Any, Dict, Optional, Union
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState

def merge_chewy_chat_state(
    old: Optional[ChewyJourneyChatState],
    new: Optional[Union[ChewyJourneyChatState, Dict[str, Any]]]) -> ChewyJourneyChatState:
    """
    Patch-merge reducer:
      - Fields present in `new` replace `old`.
      - Fields omitted in `new` are preserved from `old`.
      - Deep-merge for `debug` dict.
    """
    if old is None and new is None:
        return ChewyJourneyChatState()  # defaults

    if old is None:
        # allow dict or model
        if isinstance(new, ChewyJourneyChatState):
            return new
        return ChewyJourneyChatState.model_validate(new or {})

    if new is None:
        return old

    # Convert `new` to a patch dict that only includes explicitly set fields
    patch = new.model_dump(exclude_unset=True) if isinstance(new, ChewyJourneyChatState) else dict(new)

    # Start from old
    merged = old.model_dump()

    # Replace-or-preserve for scalar fields
    for field in ["last_user_message", "confidence", "route", "agent_response", "grounded_score", "escalate"]:
        if field in patch:
            merged[field] = patch[field]

    # Deep-merge debug
    if "debug" in patch and isinstance(patch["debug"], dict):
        merged["debug"] = {**(merged.get("debug") or {}), **patch["debug"]}

    # Validate back into the model
    return ChewyJourneyChatState.model_validate(merged)
