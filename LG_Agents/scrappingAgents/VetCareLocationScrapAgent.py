"""
LangGraph + LLM agent: Discover Chewy Vet Care ("Chewy pet vet") *location* page links

What it does
------------
- Starts at the Chewy Vet Care hub (seed defaults to https://www.chewy.com/vet-care/)
- Fetches pages within chewy.com and extracts anchors
- **Uses an LLM** to classify anchors into: `location_links` (individual clinics) vs `crawl_next` (index/sub-hub pages)
- Produces a deduped list of location URLs with a simple `name_guess` and `source`
- Built around a **Pydantic state object** so you can extend it later (e.g., add hours/phone scraping in a second agent)

How to run
----------
1) Install deps
   pip install -U langgraph langchain-core langchain-openai httpx beautifulsoup4 lxml typer rich pydantic

2) Set your OpenAI key (or replace the LLM block with your provider)
   export OPENAI_API_KEY=sk-...

3) Run (basic)
   python chewy_pet_vet_locations_langgraph_llm.py

4) Run (custom seed / model / outputs)
   python chewy_pet_vet_locations_langgraph_llm.py \
     --seed https://www.chewy.com/vet-care/ \
     --model gpt-4o-mini \
     --csv chewy_vet_locations.csv \
     --json chewy_vet_locations.json

Notes
-----
- The LLM sees only distilled anchor data (text + absolute URL) to keep tokens low.
- URL patterns change; the classifier uses heuristics + instructions rather than hardcoding routes.
- Extend `LocationDiscoveryState` freely; the graph will merge fields using the specified reducers.
"""
from __future__ import annotations

import re
import json
import time
import csv
import typing as t
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
import typer
from rich import print
from typing_extensions import Annotated
from operator import add, or_ as or_merge


from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ChewyScrapState import LocationDiscoveryState, ConfigModel, LocationRecord, LinkMiningOutput

# -----------------------------
# Helpers
# -----------------------------

def _canonicalize(url: str) -> str:
    p = urlparse(url)
    p = p._replace(query="", fragment="", netloc=p.netloc.lower())
    return urlunparse(p)


def _is_same_domain(url: str, domain: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith(domain)
    except Exception:
        return False


def _fetch(url: str, timeout_s: float = 25.0) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    last_exc = None
    for attempt in range(4):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.text
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Fetch failed for {url}: {last_exc}")


def _extract_anchors(base_url: str, html: str, only_vetcare_paths: bool = True) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    anchors = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith("javascript:"):
            continue
        abs_url = urljoin(base_url, href)
        if only_vetcare_paths and "/vet-care/" not in abs_url:
            continue
        text = (a.get_text(" ", strip=True) or "").strip()
        anchors.append({"href": _canonicalize(abs_url), "text": text})
        if len(anchors) >= 800:  # cap to keep LLM tokens in check
            break
    return anchors


# -----------------------------
# LLM node
# -----------------------------

PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise web link classifier helping find INDIVIDUAL CLINIC/LOCATION pages for "
        "Chewy Vet Care on chewy.com. You MUST output structured JSON.\n\n"
        "Guidelines:\n"
        "- 'Location' pages typically describe a single clinic with its own address/maps/booking.\n"
        "- 'Crawl next' links are index pages (lists of clinics), sub-hubs (city/state), or general marketing pages.\n"
        "- Prefer URLs under /vet-care/. Ignore non-chewy domains.\n"
        "- Normalize/return absolute URLs only.\n"
        "- Provide a short name guess based on the URL slug or anchor text when possible.\n"
    )),
    ("human", (
        "Page URL: {page_url}\n"
        "Allowed domain: {allowed_domain}\n\n"
        "Here are anchors from the page (href + visible text). Classify which ones are INDIVIDUAL location pages,\n"
        "and which ones we should crawl next to find more locations. Return structured fields.\n\n"
        "ANCHORS JSON (list of objects {{href, text}}):\n{anchors_json}\n"
    )),
])


def llm_classify_links(state: LocationDiscoveryState) -> LocationDiscoveryState:
    if not state.current_url or not state.anchors:
        return state

    llm = ChatOpenAI(model=state.config.model, temperature=state.config.temperature)
    structured_llm = llm.with_structured_output(LinkMiningOutput)

    anchors_json = json.dumps(state.anchors[:800], ensure_ascii=False)
    msg = PROMPT.format(
        page_url=state.current_url,
        allowed_domain=state.config.allowed_domain,
        anchors_json=anchors_json,
    )

    out: LinkMiningOutput = structured_llm.invoke(msg)

    # Merge into results and queue
    new_results: list[LocationRecord] = []
    for item in out.location_links:
        if not _is_same_domain(item.url, state.config.allowed_domain):
            continue
        u = _canonicalize(item.url)
        new_results.append(LocationRecord(url=u, name_guess=item.name_guess, source=state.current_url))

    new_queue = []
    for u in out.crawl_next:
        if not _is_same_domain(u, state.config.allowed_domain):
            continue
        cu = _canonicalize(u)
        if state.config.only_vetcare_paths and "/vet-care/" not in cu:
            continue
        new_queue.append(cu)

    notes = out.notes or ""
    return state.model_construct(
        results=[*state.results, *new_results],
        candidates=[*state.candidates, *[r.url for r in new_results]],
        queue=[*state.queue, *new_queue],
        llm_notes=[*state.llm_notes, f"{state.current_url}: {notes}" if notes else ""],
    )


# -----------------------------
# Graph nodes (non-LLM)
# -----------------------------

def bootstrap(state: LocationDiscoveryState) -> LocationDiscoveryState:
    q = state.queue or [state.config.seed]
    return state.model_construct(queue=q, visited=set(state.visited), current_url=None, html=None, anchors=[])


def next_url(state: LocationDiscoveryState) -> LocationDiscoveryState:
    queue = list(dict.fromkeys(state.queue))
    visited = set(state.visited)

    current = None
    while queue:
        candidate = queue.pop(0)
        if candidate in visited:
            continue
        if not _is_same_domain(candidate, state.config.allowed_domain):
            continue
        current = candidate
        break

    return state.model_construct(queue=queue, current_url=current)


def fetch_node(state: LocationDiscoveryState) -> LocationDiscoveryState:
    url = state.current_url
    if not url:
        return state
    try:
        html = _fetch(url)
        visited = set(state.visited)
        visited.add(url)
        return state.model_construct(html=html, visited=visited)
    except Exception as e:
        return state.model_construct(errors=[*state.errors, f"ERROR fetching {url}: {e}"], html=None)


def extract_anchors_node(state: LocationDiscoveryState) -> LocationDiscoveryState:
    url, html = state.current_url, state.html
    if not url or not html:
        return state
    anchors = _extract_anchors(url, html, only_vetcare_paths=state.config.only_vetcare_paths)
    return state.model_construct(anchors=anchors)


def integrate_and_dedupe(state: LocationDiscoveryState) -> LocationDiscoveryState:
    # Deduplicate results by URL and queue by URL
    dedup_results = []
    seen_u = set()
    for r in state.results:
        if r.url not in seen_u:
            seen_u.add(r.url)
            dedup_results.append(r)

    queue = list(dict.fromkeys(state.queue))

    return state.model_construct(results=dedup_results, queue=queue)


def should_continue(state: LocationDiscoveryState) -> str:
    if not state.queue:
        return "stop"
    if len(state.visited) >= state.config.max_pages:
        return "stop"
    return "loop"


# -----------------------------
# Build the graph
# -----------------------------

def build_graph() -> t.Any:
    graph = StateGraph(LocationDiscoveryState)

    graph.add_node("bootstrap", bootstrap)
    graph.add_node("next_url", next_url)
    graph.add_node("fetch", fetch_node)
    graph.add_node("extract_anchors", extract_anchors_node)
    graph.add_node("llm_classify", llm_classify_links)
    graph.add_node("integrate", integrate_and_dedupe)

    graph.set_entry_point("bootstrap")
    graph.add_edge("bootstrap", "next_url")
    graph.add_edge("next_url", "fetch")
    graph.add_edge("fetch", "extract_anchors")
    graph.add_edge("extract_anchors", "llm_classify")
    graph.add_edge("llm_classify", "integrate")

    graph.add_conditional_edges(
        "integrate",
        should_continue,
        {"loop": "next_url", "stop": END},
    )

    return graph.compile()


# -----------------------------
# CLI
# -----------------------------
app = typer.Typer(add_completion=False)


@app.command()
def run(
    seed: str = typer.Option(DEFAULT_SEED, help="Seed URL (main Chewy Vet Care page)"),
    model: str = typer.Option(DEFAULT_MODEL, help="LLM model name (OpenAI)"),
    temperature: float = typer.Option(0.0, help="LLM temperature"),
    max_pages: int = typer.Option(10, help="Page fetch budget for short crawl"),
    csv_out: t.Optional[str] = typer.Option(None, "--csv", help="Write results to CSV at this path"),
    json_out: t.Optional[str] = typer.Option(None, "--json", help="Write results to JSON at this path"),
):
    state = LocationDiscoveryState(
        config=ConfigModel(seed=seed, model=model, temperature=temperature, max_pages=max_pages),
    )

    graph = build_graph()
    final_state: LocationDiscoveryState = graph.invoke(state)

    # Print summary
    print(f"[bold green]Visited {len(final_state.visited)} pages[/bold green]")
    print(f"[bold green]Found {len(final_state.results)} location pages[/bold green]")
    for r in final_state.results:
        guess = f"  [dim]{r.name_guess}[/dim]" if r.name_guess else ""
        print(f"• {r.url}{guess}")

    # Save
    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "name_guess", "source"])
            writer.writeheader()
            writer.writerows([r.model_dump() for r in final_state.results])
        print(f"CSV written to {csv_out}")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in final_state.results], f, indent=2, ensure_ascii=False)
        print(f"JSON written to {json_out}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        print("\nInterrupted.")
