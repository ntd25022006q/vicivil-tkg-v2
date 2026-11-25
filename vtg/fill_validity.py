"""fill_validity: auto-resolve validity/issued dates for needs_review documents
mined from their own text ("có hiệu lực từ ngày ...", "ngày ... tháng ... năm ... đã ban hành").
Re-downloads the PDF from the watcher registry (text was not persisted), extracts,
regex-mines dates, updates integrated/nodes_legal_new.jsonl (staging) with provenance.
NEVER guesses: no date found -> stays needs_review.
"""
import json
import os
import re
import urllib.request
from datetime import datetime

from .watch import get, extract_text_from_pdf
from .textops import fold_text

HOME = os.path.expanduser("~/vicivil_tkg")
REG = os.path.join(HOME, "watch_registry.json")
NEW = os.path.join(HOME, "graph", "integrated", "nodes_legal_new.jsonl")

_RE_EFF = re.compile(r"(?:có hiệu lực|hiệu lực)(?: thi hành)?(?: kể từ| từ)?\s*(?:ngày)?\s*"
                     r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", re.I)
_RE_EFF_WORDS = re.compile(r"(?:có hiệu lực|hiệu lực)(?: thi hành)?(?: kể từ| từ)?\s*ngày\s*"
                           r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", re.I)
_RE_PROMUL = re.compile(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})")
# extension-type decisions: "gia hạn / tiếp tục có hiệu lực ... đến ngày D/M/Y" -> valid_to
_RE_EXT_TO = re.compile(r"(?:gia hạn|tiếp tục có hiệu lực|kéo dài hiệu lực)(?:[^.]{0,160}?)"
                        r"đến ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", re.I)
_RE_EXT_TO_N = re.compile(r"(?:gia hạn|tiếp tục có hiệu lực|kéo dài hiệu lực)(?:[^.]{0,160}?)"
                          r"đến ngày\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", re.I)
_RE_EXT_FROM = re.compile(r"(?:gia hạn|tiếp tục có hiệu lực)(?:[^.]{0,160}?)"
                          r"(?:kể từ|từ) ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", re.I)


def iso(d, m, y):
    try:
        return datetime(int(y), int(m), int(d)).date().isoformat()
    except Exception:
        return None


def main():
    reg = json.load(open(REG, encoding="utf-8"))
    rows = [json.loads(l) for l in open(NEW, encoding="utf-8")] if os.path.exists(NEW) else []
    updated = 0
    for n in rows:
        if n.get("valid_from"):
            continue
        # find registry entry by so_hieu
        so = n.get("so_hieu", "")
        entry = next((v for v in reg["docs"].values() if v.get("so_hieu") == so), None)
        if not entry or not entry.get("pdf_url"):
            continue
        raw = get(entry["pdf_url"], binary=True, tries=2, timeout=120)
        if not raw or raw[:4] != b"%PDF":
            continue
        tmp = "/tmp/vtg_fv.pdf"
        open(tmp, "wb").write(raw)
        text = extract_text_from_pdf(tmp) or ""
        ftext = fold_text(text)
        eff = None
        ext_to = None
        m = _RE_EFF_WORDS.search(text) or _RE_EFF.search(text) \
            or _RE_EFF_WORDS.search(ftext) or _RE_EFF.search(ftext)
        if m:
            eff = iso(m.group(1), m.group(2), m.group(3))
        if not eff:
            # directive convention: "có hiệu lực kể từ ngày ký" / "sau 15 ngày, kể từ ngày ký"
            # -> valid_from = promulgation date (parsed from the same text) [+ N days]
            pm = _RE_PROMUL.search(text[:3000])
            if pm and re.search(r"hiệu lực[^.]{0,80}ngày ký|ngày ký[^.]{0,80}hiệu lực", text, re.I):
                base = iso(pm.group(1), pm.group(2), pm.group(3))
                mn = re.search(r"sau\s*(\d{1,2})\s*ngày[^.]{0,60}ngày ký", text, re.I)
                if base:
                    from datetime import date, timedelta
                    d = date(int(pm.group(3)), int(pm.group(2)), int(pm.group(1)))
                    if mn:
                        d = d + timedelta(days=int(mn.group(1)))
                    eff = d.isoformat()
                    prom = base
        if not eff:
            # extension-type decisions: valid_to semantics ("gia hạn ... đến ngày ...")
            me = _RE_EXT_TO.search(text) or _RE_EXT_TO_N.search(text) or \
                 _RE_EXT_TO.search(ftext) or _RE_EXT_TO_N.search(ftext) or \
                 _RE_EXT_FROM.search(text) or _RE_EXT_FROM.search(ftext)
            if me:
                ext_to = iso(me.group(1), me.group(2), me.group(3))
                eff = ext_to  # stored separately below
        prom = None
        pm = _RE_PROMUL.search(text[:3000])
        if pm:
            prom = iso(pm.group(1), pm.group(2), pm.group(3))
        if eff:
            if ext_to:
                n["valid_to"] = eff
                n["valid_from"] = prom
                n["status"] = "validity_auto_extracted_extension"
                n["validity_prov"] = {"method": "regex-extension-clause-on-official-pdf",
                                      "pattern": "gia-han-den-ngay"}
            else:
                n["valid_from"] = eff
                n["issued"] = prom
                n["status"] = "validity_auto_extracted"
                n["validity_prov"] = {"method": "regex-on-official-pdf",
                                      "pdf_hash_len": len(raw), "pattern": "co-hieu-luc-tu-ngay"}
            updated += 1
            print(f"  [FILL] {so}: valid_from={n.get('valid_from')} valid_to={n.get('valid_to')} issued={prom}")
        else:
            print(f"  [FILL] {so}: no effective-date clause found — stays needs_review")
    with open(NEW, "w", encoding="utf-8") as f:
        for n in rows:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"[FILL] updated={updated}/{len(rows)}")


if __name__ == "__main__":
    main()
