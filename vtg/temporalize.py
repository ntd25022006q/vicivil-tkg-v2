"""temporalize: graft validity intervals onto the ENTITY layer.
Each LightRAG relationship carries prov.law_id (the document it was extracted from) —
so an entity relation inherits that document's validity window. Successor coverage
comes later via SUCCEEDS lineage; here we ground the base intervals honestly.
Output: graph/edges_entity_temporal.jsonl + stats.
"""
import json
import os

G = os.path.expanduser("~/vicivil_tkg/graph")


def jl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    docs = {n["node_id"]: n for n in jl(os.path.join(G, "nodes_legal.jsonl"))}
    edges = jl(os.path.join(G, "edges_entity.jsonl"))
    out, covered, uncovered = [], 0, 0
    for e in edges:
        law_id = (e.get("prov") or {}).get("law_id")
        d = docs.get(law_id) if law_id else None
        if d and d.get("valid_from"):
            e2 = dict(e)
            e2["valid_from"] = d["valid_from"]
            e2["valid_to"] = d.get("valid_to")
            out.append(e2)
            covered += 1
        else:
            e2 = dict(e)
            e2["valid_from"] = None
            e2["valid_to"] = None
            out.append(e2)
            uncovered += 1
    with open(os.path.join(G, "edges_entity_temporal.jsonl"), "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = {"edges": len(out), "with_interval": covered, "without_interval": uncovered,
             "coverage": round(100.0 * covered / max(1, len(out)), 1)}
    json.dump(stats, open(os.path.join(G, "temporalize_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[TEMPORALIZE]", stats)


if __name__ == "__main__":
    main()
