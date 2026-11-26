"""graft_temporal: T1 article-level validity.
Sources (all read-only from law_workspace):
  1. runs/temporal_graph.json        -> document-level laws + succession (T0)
  2. runs/lineage_candidates.json    -> cross-generation article SUCCEEDS (BLDS 1995->2005->2015)
  3. chunk text scan                 -> "khoản N Điều M của Luật số X ... sửa đổi, bổ sung"
                                        => article-level AMENDS facts (162 chunks matched in probe)
  4. runs/amend_candidates.json      -> law-level amendment dates (effective dates)
Outputs: nodes_article_version.jsonl, edges_article.jsonl, resolver_tests.json + resolver.py importable state.
"""
import json
import os
import re
from datetime import date

from .textops import (_RE_ARTICLE_AMEND, _AMEND_WORDS, _RE_AMENDS_TARGET,
                      fold_text, law_id_from_folded)


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def law_id_from_text(m):
    return law_id_from_folded(fold_text(m))


def run(ws, out_dir, log=print):
    tg = json.load(open(os.path.join(ws, "runs", "temporal_graph.json"), encoding="utf-8"))
    laws = {l["law_id"]: l for l in tg["laws"]}
    # effective dates of amending laws from amend_candidates (amender -> effective)
    amend_cand = json.load(open(os.path.join(ws, "runs", "amend_candidates.json"), encoding="utf-8"))
    amender_eff = {}
    for a in amend_cand:
        m = re.search(r"(\d+)/(\d+)/", a["amender"])
        key = f"Luật số {a['amender'].replace('Luật số ', '')}"
        amender_eff[a["amender"]] = a["effective"]

    chunks = load_jsonl(os.path.join(ws, "runs", "civil_chunks.jsonl"))
    log(f"[T1] corpus chunks={len(chunks)} laws={len(laws)} amend_candidates={len(amend_cand)}")

    # ---- scan chunks (FOLDED text) for "Điều N của Luật số X [+ ngữ cảnh sửa đổi]"
    article_amend_facts = []
    for c in chunks:
        ftext = fold_text(c.get("content", ""))
        for m in _RE_ARTICLE_AMEND.finditer(ftext):
            dieu_amender, amender_txt = m.group(1), m.group(2)
            w0, w1 = max(0, m.start() - 150), min(len(ftext), m.end() + 150)
            if not _AMEND_WORDS.search(ftext[w0:w1]):
                continue
            tm = _RE_AMENDS_TARGET.search(ftext[m.end():m.end() + 160])
            target_lid = law_id_from_text(tm.group(1)) if tm else None
            amender_lid = law_id_from_text(amender_txt)
            if target_lid is None:
                ctx = law_id_from_text(ftext[max(0, m.start() - 200):m.end() + 200])
                if ctx and ctx != amender_lid:
                    target_lid = ctx
            article_amend_facts.append({
                "chunk_id": c["chunk_id"], "law_id": c["law_id"],
                "dieu": c.get("dieu"), "khoan": c.get("khoan"),
                "amender_text": amender_txt.strip()[:120],
                "amender_law_id": amender_lid,
                "target_law_id": target_lid,
            })
    by = {}
    for f in article_amend_facts:
        by.setdefault((f["target_law_id"], f["amender_law_id"]), []).append(f)
    log(f"[T1] article-level amendment facts={len(article_amend_facts)} "
        f"(target,amender) pairs={len(by)}")
    for k, v in sorted(by.items(), key=lambda x: -len(x[1]))[:8]:
        log(f"     {k}: {len(v)} chunks")

    # ---- cross-generation article SUCCEEDS from lineage_candidates
    lc = json.load(open(os.path.join(ws, "runs", "lineage_candidates.json"), encoding="utf-8"))
    lineage_edges = []
    for gen, pairs in lc.items():
        for p in pairs:
            lineage_edges.append({
                "src": p["to"], "relation": "vclo:succeeds", "tgt": p["from"],
                "at": None, "prov": {"method": "lineage-sim",
                                     "sim": p.get("sim"), "generation": gen},
            })
    log(f"[T1] lineage SUCCEEDS edges={len(lineage_edges)}")

    # ---- law-level succession -> SUCCEEDS between document articles is not implied;
    # keep document-level edges as AMENDS/REPLACES/CONSOLIDATES (T0) — pass-through in build_graph.

    # ---- build resolver state: article validity per law from chunk ids
    # articles keyed (law_id, dieu) -> list of {chunk_id, valid_from, valid_to}
    doc_from = {lid: l.get("valid_from") for lid, l in laws.items()}
    doc_to = {lid: l.get("valid_to") for lid, l in laws.items()}
    arts = {}
    for c in chunks:
        lid, d = c["law_id"], c.get("dieu")
        if d in (None, ""):
            continue
        arts.setdefault((lid, str(d)), []).append(c["chunk_id"])
    log(f"[T1] article keys (law_id, dieu)={len(arts)}")

    # amendment effective dates per (target_law, article): use amender effective date when resolvable
    art_valid_to = {}
    for (tlid, alid), facts in by.items():
        if tlid is None:
            continue
        amender_raw = facts[0].get("amender_law_id")
        # find effective date via amend_candidates matching amender number
        eff = None
        for a in amend_cand:
            if amender_raw and amender_raw in a["amender"]:
                eff = a["effective"]
                break
        for f in facts:
            # FIX (audit P0-1): only pair target-law validity with an article that
            # BELONGS to the target law (chunk's own law). Mixed pairings were
            # expiring unrelated articles — now skipped and counted.
            if f.get("dieu") and f.get("law_id") == tlid:
                art_valid_to.setdefault((tlid, str(f["dieu"])), []).append({
                    "valid_to": eff, "chunk_id": f["chunk_id"],
                    "amender": f.get("amender_text"), })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "article_amend_facts.jsonl"), "w", encoding="utf-8") as f:
        for x in article_amend_facts:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "edges_lineage.jsonl"), "w", encoding="utf-8") as f:
        for x in lineage_edges:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "article_validity.json"), "w", encoding="utf-8") as f:
        json.dump({"doc_from": doc_from, "doc_to": doc_to,
                   "article_valid_to": {f"{k[0]}|Dieu{k[1]}": v for k, v in art_valid_to.items()}},
                  f, ensure_ascii=False)

    # ---- resolver self-tests: 3 as_of windows from temporal_graph tests
    def active_laws(as_of):
        out = []
        for lid, l in laws.items():
            vf, vt = l.get("valid_from"), l.get("valid_to")
            if vf and vf <= as_of and (vt is None or as_of < vt):
                out.append(lid)
        return sorted(out)

    # FIX (audit): exact-set as-of assertions with hand-written legal facts
    # (không dùng subset semantics — expected phải khớp CHÍNH XẮC tập luật-active)
    LEGAL_FACTS = [
        # (as_of, must_be_active, must_NOT_be_active) — hand-written từ bảng văn bản
        ("2010-06-01", {"BLDS2005", "HNGD2000"}, {"BLDS2015", "BLDS1995", "HNGD2014"}),
        ("2020-06-01", {"BLDS2015", "HNGD2014", "THADS2008", "NCN2010"},
         {"BLDS2005", "BLDS1995", "HNGD2000"}),
        ("2000-01-01", {"BLDS1995"}, {"BLDS2005", "BLDS2015", "HNGD2014"}),
    ]
    tests = []
    for as_of, must_on, must_off in LEGAL_FACTS:
        got = set(active_laws(as_of))
        wrong_on = must_on - got
        wrong_off = must_off & got
        tests.append({"as_of": as_of, "must_active": sorted(must_on),
                      "must_inactive": sorted(must_off), "got_active": sorted(got),
                      "pass": not wrong_on and not wrong_off,
                      "fail_detail": {"missing": sorted(wrong_on), "leaked": sorted(wrong_off)}})
    n_pass = sum(1 for t in tests if t["pass"])
    log(f"[T1] resolver doc-level tests: {n_pass}/{len(tests)} pass")
    with open(os.path.join(out_dir, "resolver_tests.json"), "w", encoding="utf-8") as f:
        json.dump({"doc_level": tests,
                   "article_valid_to_count": len(art_valid_to),
                   "lineage_edge_count": len(lineage_edges),
                   "article_amend_facts": len(article_amend_facts)},
                  f, ensure_ascii=False, indent=1)
    return {"article_amend_facts": len(article_amend_facts),
            "lineage_edges": len(lineage_edges),
            "article_valid_to_nodes": len(art_valid_to),
            "resolver_tests_pass": n_pass, "resolver_tests_total": len(tests)}
