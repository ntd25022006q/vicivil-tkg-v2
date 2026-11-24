"""case_wire: wire 1,200 crawled judgments into the graph via APPLIES edges.
For each judgment: find cited articles (Điều N) + mentioned laws; resolve the law to law_id;
emit APPLIES(verdict -> Law|DieuN) with as_of = verdict_date, validity-checked against doc interval.
"""
import json
import os
import re

from .textops import fold_text, law_id_from_folded, _RE_DIEU


def jl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def run(ws, out_dir, log=print, max_judgments=None):
    rj = jl(os.path.join(ws, "synapselex", "data", "raw_judgments.jsonl"))
    if max_judgments:
        rj = rj[:max_judgments]
    # doc validity for APPLIES filtering
    tg = json.load(open(os.path.join(ws, "runs", "temporal_graph.json"), encoding="utf-8"))
    dfrom = {l["law_id"]: l.get("valid_from") for l in tg["laws"]}
    dto = {l["law_id"]: l.get("valid_to") for l in tg["laws"]}

    edges, no_cite, oow = [], 0, 0
    ambiguous = 0
    coverage = 0
    for j in rj:
        text = j.get("raw_text", "") or ""
        ftext = fold_text(text)
        vdate = j.get("verdict_date")
        cites = []
        ambiguous += 1 if ("bo luat dan su" in ftext and
                           not law_id_from_folded(ftext[:ftext.find("dieu")] if "dieu" in ftext else ftext)) else 0
        for m in _RE_DIEU.finditer(ftext):
            dpos = m.start()
            before = ftext[max(0, dpos - 400):dpos]
            lid = law_id_from_folded(before)
            if lid is None:
                after = ftext[dpos:dpos + 200]
                lid = law_id_from_folded(after)
            if lid:
                cites.append((lid, m.group(1)))
        cites = list(dict.fromkeys(cites))[:12]
        if not cites:
            no_cite += 1
            continue
        n_ok = 0
        for lid, dieu in cites:
            vf, vt = dfrom.get(lid), dto.get(lid)
            if vf and vdate and vf <= vdate and (vt is None or vdate < vt):
                edges.append({
                    "src": f"CASE-{j['doc_id']}", "relation": "vclo:applies",
                    "tgt": f"{lid}|Dieu{dieu}", "at": vdate,
                    "prov": {"case_number": j.get("case_number"),
                             "court": j.get("court_name"),
                             "source_url": j.get("source_url"),
                             "as_of_type": "verdict_date"}})
                n_ok += 1
            else:
                oow += 1
        if n_ok:
            coverage += 1
    with open(os.path.join(out_dir, "edges_case.jsonl"), "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = {"judgments": len(rj), "judgments_with_applies": coverage,
             "applies_edges": len(edges), "no_citation_found": no_cite,
             "out_of_window_cites": oow,
             "yearless_law_mentions": ambiguous}
    with open(os.path.join(out_dir, "case_wire_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    log(f"[CASE] {stats}")
    return stats
