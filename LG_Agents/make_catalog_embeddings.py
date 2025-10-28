# make_catalog_embeddings_csv.py
# Stream large Chewy catalog CSV, keep every Nth row (default 100),
# create embeddings, and write JSONL with detailed pricing metadata.

from __future__ import annotations
import argparse, csv, json, os, re, sys
from typing import Dict, List, Any, Optional
from openai import OpenAI

# ---------- utilities ----------
def normkey(s: str) -> str:
    return (s or "").strip().lower().lstrip("\ufeff")

def pick(mapping: Dict[str, str], *cands: str) -> Optional[str]:
    for c in cands:
        k = normkey(c)
        if k in mapping:
            return mapping[k]
    return None

def as_money(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def as_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "": return None
        return float(x)
    except Exception:
        return None

def as_int(x: Any) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "": return None
        return int(float(x))
    except Exception:
        return None

def join_if(vals: List[str], sep: str = " > ") -> Optional[str]:
    parts = [v for v in vals if v]
    return sep.join(parts) if parts else None

_WEIGHT_RE = re.compile(r"(?i)\b(\d+(\.\d+)?)\s*(oz|ounce|ounces|lb|lbs|pound|pounds)\b")
def parse_weight(size_text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (weight_lb, weight_oz) parsed from a size string like '12-lb bag' or '3 oz pouch, pack of 12'."""
    if not size_text:
        return None, None
    m = _WEIGHT_RE.search(size_text)
    if not m:
        return None, None
    val = float(m.group(1))
    unit = m.group(3).lower()
    if unit in ("oz", "ounce", "ounces"):
        return val / 16.0, val
    # pounds
    return val, val * 16.0

def build_embed_text(prod: Dict[str, Any]) -> str:
    # Compact, information-dense text for embedding (avoid heavy numeric price emphasis).
    parts: List[str] = []
    if prod.get("title"): parts.append(prod["title"])
    line2 = []
    if prod.get("brand"): line2.append(prod["brand"])
    if prod.get("category"): line2.append(prod["category"])
    if line2: parts.append(" • ".join(line2))
    # include size flavor text if present—it helps match queries like "12-lb puppy food"
    if prod.get("size_raw"): parts.append(prod["size_raw"])
    if prod.get("description"): parts.append(prod["description"])
    return " \n".join(parts).strip()

# ---------- main ----------
def main():
    #hardcoding below for simplicity
    csv_input = "product_catelog.csv"
    output_path = "catalog_embeds.jsonl"
    embedding_model = "text-embedding-3-large"
    stride =10 #keep every 10th row
    batch_size = 128

    client = OpenAI()  # needs OPENAI_API_KEY

    # Stream headers
    with open(csv_input, newline="", encoding="utf-8-sig") as fhead:
        reader = csv.reader(fhead)
        try:
            header = next(reader)
        except StopIteration:
            print("Empty CSV.", file=sys.stderr)
            sys.exit(1)

    header_map: Dict[str, str] = {normkey(h): h for h in header}

    # Core identity columns (based on your sample)
    COL_ID    = pick(header_map, "PRODUCT_ID", "ID")
    COL_SKU   = pick(header_map, "PRODUCT_PART_NUMBER", "SKU", "PART_NUMBER")
    COL_UPC   = pick(header_map, "CASE_UPC", "UPC", "GTIN")
    COL_NAME  = pick(header_map, "PRODUCT_NAME", "NAME", "DISPLAY_NAME")
    COL_BRAND = pick(header_map, "PRODUCT_PURCHASE_BRAND", "BRAND")

    # Category columns
    COL_CATL1 = pick(header_map, "PRODUCT_CATEGORY_LEVEL1")
    COL_CATL2 = pick(header_map, "PRODUCT_CATEGORY_LEVEL2")
    COL_CATL3 = pick(header_map, "PRODUCT_CATEGORY_LEVEL3")
    COL_CATL  = pick(header_map, "PRODUCT_CATEGORY_LIST", "CATEGORY_PATH")

    # Descriptions
    COL_DESC1 = pick(header_map, "PRODUCT_DESCRIPTION_LONG")
    COL_DESC2 = pick(header_map, "PRODUCT_DESCRIPTION_SHORT")

    # Size / pack info (best-effort guesses—use whatever your file has)
    COL_SIZE  = pick(header_map, "PRODUCT_SIZE", "SIZE", "VARIANT_SIZE")
    COL_PACK  = pick(header_map, "PACK_COUNT", "CASE_PACK", "QUANTITY", "COUNT")

    # ---- Pricing: keep everything useful ----
    COL_PRICE_CURR = pick(header_map, "PRODUCT_PRICE_CURRENT", "CURRENT_PRICE", "PRICE")
    COL_PRICE_LIST = pick(header_map, "PRODUCT_PRICE_LIST", "LIST_PRICE", "MSRP")
    COL_PRICE_SALE = pick(header_map, "PRODUCT_PRICE_SALE", "SALE_PRICE", "DISCOUNT_PRICE")
    COL_PRICE_MIN  = pick(header_map, "PRODUCT_PRICE_MIN")
    COL_PRICE_MAX  = pick(header_map, "PRODUCT_PRICE_MAX")
    COL_CURRENCY   = pick(header_map, "CURRENCY", "PRICE_CURRENCY")
    COL_COST       = pick(header_map, "PRODUCT_COST", "COST")  # optional/internal; included if present
    COL_PROMO      = pick(header_map, "PROMO_FLAG", "ON_PROMO", "HAS_COUPON")
    COL_EFF_DATE   = pick(header_map, "PRICE_EFFECTIVE_DATE")

    kept = 0
    total = 0
    batch_items: List[Dict[str, Any]] = []
    batch_texts: List[str] = []

    def flush_batch():
        nonlocal kept, batch_items, batch_texts
        if not batch_texts:
            return
        resp = client.embeddings.create(model=embedding_model, input=batch_texts)
        vecs = [d.embedding for d in resp.data]
        with open(output_path, "a", encoding="utf-8") as out:
            for prod, vec in zip(batch_items, vecs):
                obj = {**{k: v for k, v in prod.items() if v not in ("", None)}, "embedding": vec}
                if not obj.get("sku") and not obj.get("id"):
                    obj["id"] = f"row-{kept:09d}"
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += len(batch_items)
        batch_items.clear()
        batch_texts.clear()

    # Stream data rows
    with open(csv_input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            total += 1
            if (i - 1) % stride != 0:
                continue

            # identity & basics
            sku   = (row.get(COL_SKU) or "").strip() if COL_SKU else None
            pid   = (row.get(COL_ID) or "").strip() if COL_ID else None
            upc   = (row.get(COL_UPC) or "").strip() if COL_UPC else None
            title = (row.get(COL_NAME) or "").strip() if COL_NAME else None
            brand = (row.get(COL_BRAND) or "").strip() if COL_BRAND else None

            if not title and not (sku or pid):
                continue

            # category path
            cat_path = join_if([
                (row.get(COL_CATL1) or "").strip() if COL_CATL1 else "",
                (row.get(COL_CATL2) or "").strip() if COL_CATL2 else "",
                (row.get(COL_CATL3) or "").strip() if COL_CATL3 else "",
            ])
            category = cat_path or ((row.get(COL_CATL) or "").strip() if COL_CATL else None)

            # descriptions
            description = (row.get(COL_DESC1) or row.get(COL_DESC2) or "").strip() if (COL_DESC1 or COL_DESC2) else None

            # size / pack
            size_raw = (row.get(COL_SIZE) or "").strip() if COL_SIZE else None
            pack_count = as_int(row.get(COL_PACK) if COL_PACK else None)
            weight_lb, weight_oz = parse_weight(size_raw or "")

            # pricing (raw)
            price_current = as_money(row.get(COL_PRICE_CURR) if COL_PRICE_CURR else None)
            price_list    = as_money(row.get(COL_PRICE_LIST) if COL_PRICE_LIST else None)
            price_sale    = as_money(row.get(COL_PRICE_SALE) if COL_PRICE_SALE else None)
            price_min     = as_money(row.get(COL_PRICE_MIN) if COL_PRICE_MIN else None)
            price_max     = as_money(row.get(COL_PRICE_MAX) if COL_PRICE_MAX else None)
            currency      = (row.get(COL_CURRENCY) or "").strip() if COL_CURRENCY else None
            cost          = as_money(row.get(COL_COST) if COL_COST else None)
            promo_flag    = (row.get(COL_PROMO) or "").strip() if COL_PROMO else None
            price_effective_date = (row.get(COL_EFF_DATE) or "").strip() if COL_EFF_DATE else None

            # pricing (derived)
            # prefer explicit sale price as "price" if present, else current, else list.
            primary_price = price_sale or price_current or price_list
            discount_pct = None
            base_for_discount = price_list or price_current
            if base_for_discount and primary_price is not None and base_for_discount > 0:
                discount_pct = round((base_for_discount - primary_price) / base_for_discount * 100.0, 2)

            price_per_lb = round(primary_price / weight_lb, 4) if primary_price and weight_lb and weight_lb > 0 else None
            price_per_oz = round(primary_price / weight_oz, 4) if primary_price and weight_oz and weight_oz > 0 else None

            prod = {
                # identity
                "id": pid or None,
                "sku": sku or None,
                "upc": upc or None,
                "title": title or None,
                "brand": brand or None,
                "category": category or None,
                "description": description or None,

                # size/pack
                "size_raw": size_raw or None,
                "pack_count": pack_count,
                "weight_lb": weight_lb,
                "weight_oz": weight_oz,

                # pricing (keep everything)
                "price": primary_price,             # backward-compat (what most agents look at)
                "price_current": price_current,
                "price_sale": price_sale,
                "price_list": price_list,
                "price_min": price_min,
                "price_max": price_max,
                "currency": currency or None,
                "discount_pct": discount_pct,
                "price_per_lb": price_per_lb,
                "price_per_oz": price_per_oz,
                "promo_flag": promo_flag or None,
                "price_effective_date": price_effective_date or None,

                # optional internal
                "cost": cost,
            }

            text = build_embed_text(prod)
            if True and primary_price is not None:
                text = f"{text}\nPrice: {primary_price} {currency or ''}".strip()
            if not text:
                continue

            batch_items.append(prod)
            batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                flush_batch()

    # flush remainder
    flush_batch()

    print(f"Done. Scanned {total} rows, kept {kept} (~{(kept/(total or 1))*100:.2f}%).")
    print(f"Wrote: {output_path}")
    print("Tip: set CATALOG_EMBEDS_PATH to point your retriever at this file.")

if __name__ == "__main__":
    main()
