# build_pet_insurance_index.py
from __future__ import annotations
import os, json, math, re, uuid, pathlib
from typing import List, Dict, Any, Optional, Tuple

# pip install python-docx tqdm
from docx import Document
from tqdm import tqdm

# Optional OpenAI embeddings
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _clean(t: str) -> str:
    t = (t or "").replace("\xa0", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    return t

def _docx_to_sections(path: str) -> List[Tuple[str, str]]:
    """
    Returns list of (section_path, paragraph_text).
    section_path is a string like "Chewy Insurance > Claims > Filing a claim".
    """
    doc = Document(path)
    sections = []
    stack: List[str] = []
    for p in doc.paragraphs:
        txt = _clean(p.text)
        if not txt:
            continue
        style = (p.style.name if p.style else "").lower()
        # Heuristic: treat Heading 1..4 as section headers
        m = re.match(r"heading\s*(\d+)", style)
        if m:
            lvl = int(m.group(1))
            # maintain a stack of headings
            while len(stack) >= lvl:
                stack.pop()
            stack.append(txt)
            continue
        # normal paragraph; attach to current section path
        sec_path = " > ".join(stack) if stack else "Document"
        sections.append((sec_path, txt))
    return sections

def _chunk_paragraphs(sections: List[Tuple[str, str]],
                      max_chars: int = 1200,
                      overlap: int = 120) -> List[Dict[str, Any]]:
    """
    Pack paragraphs into ~max_chars chunks, keeping section headings as metadata.
    """
    chunks: List[Dict[str, Any]] = []
    buf = []
    cur_sec = None
    cur_len = 0

    def flush():
        nonlocal buf, cur_sec, cur_len
        if not buf:
            return
        text = "\n".join([b[1] for b in buf]).strip()
        sec = buf[0][0] if buf else (cur_sec or "Document")
        chunks.append({
            "id": str(uuid.uuid4()),
            "section": sec,
            "text": text,
            "source": os.path.basename(INPUT_PATH),
        })
        if overlap and len(buf) > 1:
            # keep tail of buffer for overlap
            keep_text = "\n".join([b[1] for b in buf[-max(1, math.ceil(len(buf)*0.25)):]])
            buf = [(sec, keep_text)]
            cur_len = len(keep_text)
        else:
            buf, cur_len = [], 0

    for sec, para in sections:
        para = _clean(para)
        if not para:
            continue
        if cur_sec is None:
            cur_sec = sec
        if cur_len + len(para) + 1 > max_chars:
            flush()
            cur_sec = sec
        buf.append((sec, para))
        cur_len += len(para) + 1
    flush()
    return chunks

def _embed_texts_openai(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Batch-friendly: embeddings.create can take many inputs
    resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _tokenize_for_fallback(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [w for w in s.split() if len(w) > 2]

def build_index(input_path: str, out_path: str):
    sections = _docx_to_sections(input_path)
    chunks = _chunk_paragraphs(sections)

    # Add vectors if OpenAI key present; else store token sets for fallback scoring
    if OPENAI_API_KEY:
        vecs = _embed_texts_openai([c["text"] for c in tqdm(chunks, desc="Embedding")])
        for c, v in zip(chunks, vecs):
            c["vector"] = v
    else:
        for c in chunks:
            c["tokens"] = _tokenize_for_fallback(c["text"])

    pathlib.Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks → {out_path}")

if __name__ == "__main__":
    # Set these via env or edit below
    INPUT_PATH = os.getenv("CJ_INSURANCE_DOC", "C:/genAIProjects/new-pet-parent-poc/backend/LG_Agents/embeddings/chewycareplus.docx")
    OUTPUT_INDEX = os.getenv("CJ_INSURANCE_INDEX", "C:/genAIProjects/new-pet-parent-poc/backend/LG_Agents/embeddings/pet_insurance_index.jsonl")
    build_index(INPUT_PATH, OUTPUT_INDEX)
