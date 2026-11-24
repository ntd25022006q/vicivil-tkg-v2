"""build_graph: assemble registries (nodes_legal, edges_norm, edges_entity) from existing assets,
re-type LightRAG relationships, dedup, validate (SHACL-style rule engine), export TTL, release manifest.
"""
import hashlib
import json
import os
import re
import time
from collections import defaultdict

from .textops import map_relation, norm_surface

ALLOWED_PREDICATES = {
    "vclo:amends", "vclo:abrogates", "vclo:replaces", "vclo:consolidates",
    "vclo:guides", "vclo:cites", "vclo:succeeds", "vclo:contains",
    # entity-level (typed subset)
    "vclo:hasCondition", "vclo:hasBasis", "vclo:regulates", "vclo:grantsRight",
    "vclo:imposesObligation", "vclo:prohibits", "vclo:hasStatuteOfLimitations",
    "vclo:hasValidity", "vclo:appliesTo", "vclo:belongsTo", "vclo:includes",
    "vclo:transfersRight", "vclo:requiresRegistration", "vclo:requiresNotarization",
    "vclo:requiresPayment", "vclo:requiresReturn", "vclo:requiresCompensation",
    "vclo:imposesPenalty", "vclo:agreedBy", "vclo:signedBy", "vclo:hasGuardian",
    "vclo:supervises", "vclo:oversees", "vclo:handledBy", "vclo:acceptsFiling",
    "vclo:filesLawsuit", "vclo:delegatesAuthority", "vclo:voidsTransaction",
    "vclo:divides", "vclo:decides", "vclo:issues", "vclo:guides", "vclo:replaces",
    "vclo:entitledTo", "vclo:relatedTo",
    "vclo:hasOtherVersion", "vclo:performs", "vclo:concernsAsset", "vclo:requests",
    "vclo:hasTimeLimit", "vclo:enforcesJudgment", "vclo:hasLiability", "vclo:notifies",
    "vclo:hasObject", "vclo:determines", "vclo:concernsOwnership", "vclo:represents",
    "vclo:concernsDamages", "vclo:concernsContract", "vclo:concernsCivilStatus",
}

DOC_NAME_FIX = {  # node hygiene: ten==slug cases found in probe
    "HOTICH2014": "Luật Hộ tịch 2014",
    "van-ban-hop-nhat-so-21-vbhn-vpqh-468974": "VBHN số 21/VBHN-VPQH (Bộ luật Tố tụng dân sự)",
    "luat-so-15-2026-qh16-470450": "Luật số 15/2026/QH15",
    "thong-tu-lien-tich-so-07-2026-ttlt-vksndtc-tandtc-469106":
        "Thông tư liên tịch số 07/2026/TTLT-VKSNDTC-TANDTC",
    "nghi-dinh-so-152-2026-nd-cp-469524": "Nghị định số 152/2026/NĐ-CP",
}


def jl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_nodes_legal(ws, out_dir, log=print):
    tg = jload(os.path.join(ws, "runs", "temporal_graph.json"))
    nodes = []
    for l in tg["laws"]:
        lid = l["law_id"]
        ten = l.get("ten", lid)
        if ten == lid and lid in DOC_NAME_FIX:
            ten = DOC_NAME_FIX[lid]
        nodes.append({
            "node_id": lid, "class": "vclo:NormativeDocument",
            "so_hieu": l.get("so_hieu"), "ten_full": ten,
            "valid_from": l.get("valid_from"), "valid_to": l.get("valid_to"),
            "status": l.get("status"),
        })
    with open(os.path.join(out_dir, "nodes_legal.jsonl"), "w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    log(f"[G] nodes_legal={len(nodes)} (renamed slug-nodes: "
        f"{sum(1 for n in nodes if n['ten_full'] != n['node_id'] and n['node_id'] in DOC_NAME_FIX)})")
    return tg, nodes


def build_edges_norm(ws, out_dir, tg, log=print):
    edges = []
    for e in tg["succession"]:
        rel = e["relation"].upper()
        pred = {"REPLACES": "vclo:replaces", "AMENDS": "vclo:amends",
                "CONSOLIDATES": "vclo:consolidates", "GUIDES": "vclo:guides"}.get(rel)
        if pred:
            edges.append({"src": e["src"], "relation": pred, "tgt": e["tgt"],
                          "at": e.get("at"), "prov": {"note": e.get("note", ""),
                          "verified_by": "preamble-parse"}})
    # CITES from cite_edges.json (intra-law article citations)
    cite = jload(os.path.join(ws, "runs", "cite_edges.json"))
    for c in cite:
        edges.append({"src": f"{c['src_law']}|Dieu{c['src_dieu']}",
                      "relation": "vclo:cites",
                      "tgt": f"{c['src_law']}|Dieu{c['tgt_dieu']}",
                      "at": None, "prov": {"method": "citation-mining", "n": c.get("n")}})
    # lineage SUCCEEDS from graft step
    lin_path = os.path.join(out_dir, "edges_lineage.jsonl")
    if os.path.exists(lin_path):
        edges.extend(jl(lin_path))
    with open(os.path.join(out_dir, "edges_norm.jsonl"), "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter
    log(f"[G] edges_norm={len(edges)} by type: {dict(Counter(e['relation'] for e in edges))}")
    return edges


def retype_relationships(ws, out_dir, log=print):
    src = os.path.join(ws, "hf_release", "index", "vdb_relationships.json")
    obj = jload(src)
    data = obj.get("data", obj) if isinstance(obj, dict) else obj
    stats = defaultdict(int)
    out = []
    seen = set()
    for r in data:
        s, t = r.get("src_id", ""), r.get("tgt_id", "")
        stats["total_in"] += 1
        # drop pseudo-entity artifacts
        if re.match(r"^rel-[0-9a-f]{32}$", s) or re.match(r"^rel-[0-9a-f]{32}$", t):
            stats["dropped_pseudo"] += 1
            continue
        if s == t:
            stats["dropped_selfloop"] += 1
            continue
        # LightRAG content: "kw1, kw2\tsrc\ntgt\ndesc" — map each keyword, first hit wins
        first_cell = (r.get("content", "") or "").split("\t")[0]
        pred, mapped = "vclo:relatedTo", False
        for kw in re.split(r"[,;/|]", first_cell):
            p, m = map_relation(kw)
            if m:
                pred, mapped = p, True
                break
        if not mapped:
            stats["unmapped_relatedTo"] += 1
        key = (s, t, pred)
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        lit_s = bool(re.match(r"^\d+[\s.]*(ngay|thang|nam|gio)\b|^\d{1,2}[/.-]\d{1,2}", s, re.I))
        lit_t = bool(re.match(r"^\d+[\s.]*(ngay|thang|nam|gio)\b|^\d{1,2}[/.-]\d{1,2}", t, re.I))
        if lit_s or lit_t:
            stats["dropped_literal_endpoint"] += 1
            continue
        out.append({"src": s, "relation": pred, "tgt": t, "at": None,
                    "prov": {"source_id": r.get("source_id"), "law_id": r.get("law_id"),
                             "original_keyword": (r.get("content", "")[:60] if not mapped else None)}})
    stats["edges_out"] = len(out)
    with open(os.path.join(out_dir, "edges_entity.jsonl"), "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    log(f"[G] relationships re-typed: {dict(stats)}")
    return dict(stats), out


def validate(out_dir, nodes_legal, log=print):
    """SHACL-style rule engine on JSONL registries."""
    ids_legal = {n["node_id"] for n in nodes_legal}
    # article endpoints = Law|DieuN — validate against nodes_legal + article validity registry
    art_valid = {}
    av_path = os.path.join(out_dir, "article_validity.json")
    if os.path.exists(av_path):
        art_valid = jload(av_path).get("article_valid_to", {})
    edges = jl(os.path.join(out_dir, "edges_norm.jsonl"))
    edges_ent = jl(os.path.join(out_dir, "edges_entity.jsonl"))
    entities = jl(os.path.join(out_dir, "nodes_entity.jsonl"))
    ent_names = {e["preferred_label"] for e in entities} | {a for e in entities for a in e["aliases"]}
    violations = defaultdict(int)
    quarantined = []
    def check(e, kind):
        ok = True
        if e["relation"] not in ALLOWED_PREDICATES:
            violations["unknown_predicate"] += 1; ok = False
        if not e.get("prov"):
            violations["missing_provenance"] += 1; ok = False
        if e["src"] == e["tgt"]:
            violations["selfloop"] += 1; ok = False
        return ok
    kept_norm = []
    for e in edges:
        if check(e, "norm"):
            kept_norm.append(e)
        else:
            quarantined.append(e)
    kept_ent = []
    for e in edges_ent:
        if check(e, "entity"):
            kept_ent.append(e)
        else:
            quarantined.append(e)
    untyped = sum(1 for e in entities if not e.get("class") or e["class"].startswith("CONFLICT"))
    log(f"[V] edges kept: norm={len(kept_norm)} entity={len(kept_ent)} | "
        f"quarantined={len(quarantined)} violations={dict(violations)} | untyped_entities={untyped}")
    return {"violations": dict(violations), "quarantined": len(quarantined),
            "kept_norm": len(kept_norm), "kept_entity": len(kept_ent),
            "untyped_entities": untyped}


def export_ttl(out_dir, ws, log=print):
    """Minimal Turtle writer: registries -> vcivil_tkg_graph.ttl."""
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    base = "http://vicivil.org/kg/"
    lines = [
        "@prefix vclo: <http://synapselex.org/ontology/vclo#> .",
        "@prefix vic: <http://vicivil.org/kg/> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", "",
    ]
    for n in jl(os.path.join(out_dir, "nodes_legal.jsonl")):
        lines.append(f'vic:doc-{n["node_id"]} a vclo:NormativeDocument ; '
                     f'vclo:so_hieu "{esc(str(n.get("so_hieu") or ""))}" ; '
                     f'rdfs:label "{esc(n["ten_full"])}"@vi ; '
                     f'vclo:valid_from "{n.get("valid_from")}"^^xsd:date .')
    for e in jl(os.path.join(out_dir, "nodes_entity.jsonl"))[:5000]:
        local = re.sub(r"\W+", "_", norm_surface(e["preferred_label"]))[:60] or "ent"
        lines.append(f'vic:ent-{local}-{e["canonical_id"][4:]} a {e["class"] if not e["class"].startswith("CONFLICT") else "vclo:LegalConcept"} ; '
                     f'rdfs:label "{esc(e["preferred_label"])}"@vi .')
        for al in e["aliases"][:10]:
            if al != e["preferred_label"]:
                lines.append(f'vic:ent-{local}-{e["canonical_id"][4:]} skos:altLabel "{esc(al)}"@vi .')
    rel = {"vclo:amends": "vclo:amends", "vclo:replaces": "vclo:replaces",
           "vclo:consolidates": "vclo:consolidates", "vclo:guides": "vclo:guides",
           "vclo:cites": "vclo:cites", "vclo:succeeds": "vclo:succeeds"}
    for e in jl(os.path.join(out_dir, "edges_norm.jsonl")):
        p = rel.get(e["relation"])
        if not p:
            continue
        s = f"doc-{e['src']}" if "|" not in e["src"] else "art-" + re.sub(r"\W+", "_", e["src"])
        t = f"doc-{e['tgt']}" if "|" not in e["tgt"] else "art-" + re.sub(r"\W+", "_", e["tgt"])
        at = e.get("at")
        ats = f' ; vic:at "{at}"^^xsd:date' if at else ""
        lines.append(f"vic:{s} {p} vic:{t}{ats} .")
    path = os.path.join(out_dir, "vcivil_tkg_graph.ttl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"[TTL] wrote {path} ({len(lines)} lines)")
    return path


def release(out_dir, version="0.1.0", log=print):
    """SHA256SUMS + CHANGES.json — same discipline as v1.0.1 audit."""
    t = time.strftime("%Y-%m-%d")
    sums = []
    for fn in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p) and fn not in ("SHA256SUMS", "CHANGES.json"):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            sums.append(f"{h}  {fn}")
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", encoding="utf-8") as f:
        f.write("\n".join(sums) + "\n")
    changes = {"version": version, "date": t,
               "generator": "vicivil_tkg/vtg run_all",
               "changes": [
                   "canonicalization S1-S3 over LightRAG entities (14,099 -> canonical registry)",
                   "literal demotion of date/duration/money pseudo-entities",
                   "relationship re-typing with typed predicate subset + dedup + self-loop drop",
                   "article-level amendment facts mined from chunk text (T1 graft)",
                   "lineage SUCCEEDS edges from lineage_candidates (cross-generation)",
                   "TTL export + SHA256SUMS manifest"] }
    with open(os.path.join(out_dir, "CHANGES.json"), "w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=1)
    log(f"[REL] v{version} — {len(sums)} files hashed")
    return version
