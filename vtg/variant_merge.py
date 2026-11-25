"""variant_merge: dual-support auto-approval of SPELLING-VARIANT pairs (B3 doctrine).
A rejected/pending pair is auto-approved for merge ONLY when BOTH gates pass:
  gate 1 (text): Vietnamese spelling-variant equality — fold + i/y final-syllable unification
                 (quí/quý, …) — NOT plain substring similarity;
  gate 2 (topology): the two entities share at least one source chunk_id (dual support).
Output: graph/approved_pairs.jsonl (consumed by canonicalize.run as tier0 merges).
"""
import json
import os
import re

from .textops import fold_text

G = os.path.expanduser("~/vicivil_tkg/graph")
WS = os.path.expanduser("~/law_workspace")


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fold_iy(s: str) -> str:
    """fold_text + unify final-syllable i/y ('quý'/'quí' → 'qui')."""
    toks = fold_text(s).split()
    out = []
    for t in toks:
        if len(t) >= 3 and t.endswith("y") and t[-2] in "aeiou":
            t = t[:-1] + "i"
        out.append(t)
    return " ".join(out)


def main():
    vdb = json.load(open(os.path.join(WS, "hf_release", "index", "vdb_entities.json"), encoding="utf-8"))
    data = vdb.get("data", vdb) if isinstance(vdb, dict) else vdb
    by_name = {e.get("entity_name", ""): e for e in data}

    # candidate rejected pairs: tier3 rejections in merge_log + pending_adjudicated decisions
    cand = []
    for m in jl(os.path.join(G, "merge_log.jsonl")):
        if m["method"] == "tier3-llm-rejected":
            names = (m.get("evidence") or {}).get("names") or []
            if len(names) == 2:
                cand.append(tuple(names))
    for p in jl(os.path.join(G, "pending_adjudicated.jsonl")):
        if p.get("decision") == "rejected":
            cand.append((p["name_a"], p["name_b"]))
    cand = list(dict.fromkeys(cand))
    print(f"[VARIANT] rejected-pair candidates: {len(cand)}")

    approved, stats = [], {"spelling_variant": 0, "with_topology": 0, "no_topology": 0}
    for na, nb in cand:
        ea, eb = by_name.get(na), by_name.get(nb)
        if not ea or not eb:
            continue
        if fold_iy(na) != fold_iy(nb):
            continue
        stats["spelling_variant"] += 1
        shared = set(ea.get("chunk_ids") or []) & set(eb.get("chunk_ids") or [])
        if shared:
            stats["with_topology"] += 1
            approved.append({"name_a": na, "name_b": nb,
                             "reason": f"spelling-variant + shared-chunk topology ({len(shared)} chunks)",
                             "confidence": 0.95})
        else:
            stats["no_topology"] += 1
    with open(os.path.join(G, "approved_pairs.jsonl"), "w", encoding="utf-8") as f:
        for a in approved:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print("[VARIANT]", stats, f"-> approved_pairs.jsonl: {len(approved)}")


if __name__ == "__main__":
    main()
