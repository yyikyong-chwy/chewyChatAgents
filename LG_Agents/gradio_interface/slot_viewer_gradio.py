# slot_viewer_gradio.py
# Gradio UI to:
# 1) Select a customer ID and run the LangGraph pipeline in npp_agents_orchestrator.
# 2) Browse slots and their candidates in a clean grid.
# 3) Load a previously saved JSON state to explore without re-running.

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import html
import sys
import gradio as gr

# ---------------------------
# Imports: be flexible about repo layout
# ---------------------------
SessionStateModel = None
save_state = None
load_state = None
load_langgraph_state_session = None
build_graph = None

from LG_Agents.states.sessionState import SessionStateModel, save_state, load_state
from LG_Agents.states.load_customer_Session import load_langgraph_state_session
from LG_Agents.bundlerAgents.npp_agents_orchestrator import build_graph

# ---------------------------
# UI helpers
# ---------------------------

_INLINE_STYLE = """
<style>
  .title{margin:0 0 .5rem 0}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
  .card{border:1px solid #e5e7eb;border-radius:14px;padding:12px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
  .slot-header{display:flex;align-items:flex-start;gap:12px;margin:4px 0 12px 0}
  .slot-title{font-weight:700;font-size:1.1rem}
  .slot-sub{color:#6b7280;margin-top:2px}
  .slot-reason{margin-top:6px;color:#374151;font-size:.95rem}
  .prod-title{font-weight:700;margin-bottom:6px}
  .meta-row{display:flex;justify-content:space-between;gap:8px;font-size:.95rem;margin:.1rem 0}
  .meta{color:#6b7280}
  .scores{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0}
  .score-label{color:#6b7280;margin-right:6px}
  .score-val{font-weight:700}
  .reasoning{margin-top:6px;font-size:.95rem;color:#111827}
  .sku{margin-top:8px;color:#6b7280;font-size:.85rem}
</style>
"""

def _as_model(x: Any) -> SessionStateModel:
    """Coerce a dict/obj into SessionStateModel."""
    if isinstance(x, SessionStateModel):
        return x
    return SessionStateModel.model_validate(x)

def _truncate(s: Optional[str], n: int = 220) -> str:
    if not s:
        return ""
    s2 = s.strip()
    return s2 if len(s2) <= n else s2[: n - 1] + "…"

def _slot_choices(state: SessionStateModel) -> List[str]:
    """Return slot_id list for the dropdown."""
    slots = getattr(state.bundle_state, "slots", None) or []
    return [s.slot_id for s in slots]

def _format_overview(state: SessionStateModel) -> str:
    c = state.customer
    p = state.pet_profile
    lines = [
        f"<h3 class='title'>Customer & Pet Overview</h3>",
        "<div class='grid'>",
        f"<div class='card'><div class='label'>Customer</div><div>{html.escape(str(getattr(c,'first_name', '') or ''))} {html.escape(str(getattr(c,'last_name','') or ''))} (ID {html.escape(str(getattr(c,'id','') or ''))})</div></div>",
        f"<div class='card'><div class='label'>Email</div><div>{html.escape(str(getattr(c,'email','') or '—'))}</div></div>",
        f"<div class='card'><div class='label'>ZIP</div><div>{html.escape(str(getattr(c,'zip_code','') or '—'))}</div></div>",
        f"<div class='card'><div class='label'>Pet</div><div>{html.escape(str(getattr(p,'pet_name','') or 'Pet'))} ({html.escape(str(getattr(p,'species','') or '—'))}) • {getattr(p,'age_months', '—')} mo • {getattr(p,'weight_lb','—')} lb</div></div>",
        f"<div class='card'><div class='label'>Breed</div><div>{html.escape(str(getattr(p,'breed','') or '—'))}</div></div>",
        f"<div class='card'><div class='label'>Signals</div><div>Habits: {', '.join(getattr(p,'habits',[]) or []) or '—'}<br>Conditions: {', '.join(getattr(p,'recent_conditions',[]) or []) or '—'}<br>Geo: {', '.join(getattr(p,'geo_eventcondition',[]) or []) or '—'}</div></div>",
        "</div>",
        _INLINE_STYLE,
    ]
    return "\n".join(lines)

def _slot_grid_html(state: SessionStateModel, slot_id: Optional[str]) -> str:
    if not slot_id:
        return "<div>No slot selected.</div>" + _INLINE_STYLE

    slots = getattr(state.bundle_state, "slots", None) or []
    slot = next((s for s in slots if s.slot_id == slot_id), None)
    if slot is None:
        return "<div>No such slot.</div>" + _INLINE_STYLE

    header = f"""
    <div class='slot-header'>
      <div>
        <div class='slot-title'>{html.escape(slot.name or slot.slot_id)}</div>
        <div class='slot-sub'>
            product_type: {html.escape(slot.product_type or '—')} · role: {html.escape(slot.role)}
        </div>
        <div class='slot-reason'>{html.escape(slot.reason_for_suggestion or '—')}</div>
      </div>
    </div>
    """

    cards = []
    for c in getattr(slot, "candidates", []) or []:
        title = html.escape(getattr(c, "title", None) or f"SKU {getattr(c, 'sku', '')}")
        brand = html.escape(getattr(c, "brand", None) or "—")
        cat = html.escape(getattr(c, "category", None) or "—")
        price_val = getattr(c, "price", None)
        price = "—" if price_val is None else f"${price_val:,.2f}"
        vec = getattr(c, "score_vector", None)
        fused = getattr(c, "score_fused", None)
        rerank = getattr(c, "score_rerank", None)
        reasoning = html.escape(_truncate(getattr(c, "reasoning", None), 300) or "")
        sku = html.escape(str(getattr(c, "sku", "")))
        vec_s = "—" if vec is None else f"{float(vec):.3f}"
        fused_s = "—" if fused is None else f"{float(fused):.3f}"
        rerank_s = "—" if rerank is None else f"{float(rerank):.3f}"

        card = f"""
        <div class='card'>
          <div class='prod-title'>{title}</div>
          <div class='meta-row'><span class='meta'>Brand</span><span>{brand}</span></div>
          <div class='meta-row'><span class='meta'>Category</span><span>{cat}</span></div>
          <div class='meta-row'><span class='meta'>Price</span><span>{price}</span></div>
          <div class='scores'>
            <div><span class='score-label'>rerank</span><span class='score-val'>{rerank_s}</span></div>
            <div><span class='score-label'>fused</span><span class='score-val'>{fused_s}</span></div>
            <div><span class='score-label'>vector</span><span class='score-val'>{vec_s}</span></div>
          </div>
          <div class='reasoning'>{reasoning}</div>
          <div class='sku'>SKU: {sku}</div>
        </div>
        """
        cards.append(card)

    if not cards:
        cards.append("<div class='card'>No candidates for this slot.</div>")

    return header + "<div class='grid'>" + "\n".join(cards) + "</div>" + _INLINE_STYLE

def _bundle_rows(state: SessionStateModel) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    b = getattr(state.bundle_state, "proposed_bundle", None)
    items = getattr(b, "items", None) or []
    for i in items:
        rows.append(
            {
                "slot_id": getattr(i, "slot_id", ""),
                "slot_name": getattr(i, "slot_name", ""),
                "sku": getattr(i, "sku", ""),
                "title": getattr(i, "title", ""),
                "brand": getattr(i, "brand", ""),
                "category": getattr(i, "category", ""),
                "price": getattr(i, "price", None),
                "score": getattr(i, "score", None),
                "score_type": getattr(i, "score_type", ""),
                "reasoning": getattr(i, "reasoning", ""),
            }
        )
    return rows

# ---------------------------
# Core actions used by events
# ---------------------------

def run_pipeline(customer_id: str | int):
    """
    Build start state for the given customer, run the compiled LangGraph, and
    return everything the UI needs. IMPORTANT: atomically update the dropdown.
    """
    # Build initial state for this customer
    state0 = load_langgraph_state_session(int(customer_id))
    # Compile + run graph
    app = build_graph()
    out = app.invoke(state0)  # May return dict; coerce below
    model = _as_model(out)

    # Dropdown choices & first value
    slot_ids = _slot_choices(model)
    first_slot_id = slot_ids[0] if slot_ids else None
    slot_dd_update = gr.update(choices=slot_ids, value=first_slot_id)

    overview_html = _format_overview(model)
    grid_html = _slot_grid_html(model, first_slot_id) if first_slot_id else "<div>No slots.</div>" + _INLINE_STYLE
    bundle = _bundle_rows(model)

    # Return order matches .click outputs below
    return model.model_dump(), slot_dd_update, grid_html, bundle, overview_html

def load_state_file(file_obj):
    """
    Load previously saved JSON state and prep UI. Atomically update dropdown.
    """
    path = Path(file_obj.name)
    model = load_state(path)

    slot_ids = _slot_choices(model)
    first_slot_id = slot_ids[0] if slot_ids else None
    slot_dd_update = gr.update(choices=slot_ids, value=first_slot_id)

    overview_html = _format_overview(model)
    grid_html = _slot_grid_html(model, first_slot_id) if first_slot_id else "<div>No slots.</div>" + _INLINE_STYLE
    bundle = _bundle_rows(model)

    return model.model_dump(), slot_dd_update, grid_html, bundle, overview_html

def render_slot_from_state(state_json: dict, slot_id: Optional[str]) -> str:
    model = _as_model(state_json)
    return _slot_grid_html(model, slot_id)

def save_state_file(state_json: dict, base_filename: str) -> str:
    if not base_filename:
        base_filename = "last_session.json"
    base = Path(base_filename)
    model = _as_model(state_json)
    save_state(model, base)
    return f"Saved to {base.resolve()}"

# ---------------------------
# Gradio layout & wiring
# ---------------------------

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# 🐾 New Pet Parent — Slot & Bundle Viewer")

    # Holds dict form of SessionStateModel between events
    state_store = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            customer_id = gr.Dropdown(
                choices=[("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6", "6")],
                value="1",
                label="Customer ID",
            )
            run_btn = gr.Button("▶️ Run LangGraph for Customer")

            load_box = gr.File(label="Load previously saved state (.json)", file_types=[".json"])
            save_name = gr.Textbox(label="Save as (filename.json)", placeholder="last_session.json")
            save_btn = gr.Button("💾 Save Current State")

            status = gr.Markdown("")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Overview"):
                    overview_md = gr.HTML()
                with gr.Tab("Slots & Candidates"):
                    slot_picker = gr.Dropdown(label="Select slot", choices=[], value=None)
                    grid_html = gr.HTML()
                with gr.Tab("Final Bundle"):
                    bundle_table = gr.Dataframe(
                        headers=[
                            "slot_id",
                            "slot_name",
                            "sku",
                            "title",
                            "brand",
                            "category",
                            "price",
                            "score",
                            "score_type",
                            "reasoning",
                        ],
                        datatype=["str", "str", "str", "str", "str", "str", "number", "number", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )

    # Events — IMPORTANT: dropdown is updated atomically with gr.update(choices=..., value=...)
    run_btn.click(
        fn=run_pipeline,
        inputs=[customer_id],
        outputs=[state_store, slot_picker, grid_html, bundle_table, overview_md],
    )

    load_box.change(
        fn=load_state_file,
        inputs=[load_box],
        outputs=[state_store, slot_picker, grid_html, bundle_table, overview_md],
    )

    slot_picker.change(
        fn=render_slot_from_state,
        inputs=[state_store, slot_picker],
        outputs=[grid_html],
    )

    save_btn.click(
        fn=save_state_file,
        inputs=[state_store, save_name],
        outputs=[status],
    )

if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]  # ...\LG_Agents
    sys.path.insert(0, str(ROOT))

    demo.launch()
