"""topology_audit: B3 dual-support confirmation — audit every accepted tier2/tier3 merge
for TOPOLOGY support (shared chunk_ids / shared law_scope between merged names).
Merge WITHOUT topology support -> flagged 'needs_topology_review' (no registry mutation).
Output: graph/merge_topology_audit.jsonl + stats.
"""
import json
import os
from collections import defaultdict

G = os.path.expanduser("~/vicivil_tkg/graph")


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ents = jl(os.path.join(G, "nodes_entity.jsonl"))
    by_alias = {}
    for e in ents:
        for a in e["aliases"]:
            by_alias[a] = e
        by_alias[e["preferred_label"]] = e
    adj = defaultdict(set)  # canonical_id -> set of chunk_ids
    for e in ents:
        adj[e["canonical_id"]] = set(e.get("chunk_ids") or [])

    rows, stats = [], {"audited": 0, "topology_supported": 0, "flagged": 0, "unresolvable": 0}
    for m in jl(os.path.join(G, "merge_log.jsonl")):
        if not m["method"].startswith(("tier2", "tier3")):
            continue
        names = (m.get("evidence") or {}).get("names") or []
        ea = by_alias.get(names[0] if names else None)
        eb = by_alias.get(names[1]) if len(names) > 1 else None
        stats["audited"] += 1
        if not ea or not eb:
            stats["unresolvable"] += 1
            rows.append({**m, "topology": "unresolvable"})
            continue
        shared_chunks = adj[ea["canonical_id"]] & adj[eb["canonical_id"]]
        shared_scope = set(ea.get("law_scope") or []) & set(eb.get("law_scope") or [])
        supported = bool(shared_chunks) or bool(shared_scope)
        stats["topology_supported" if supported else "flagged"] += 1
        rows.append({**m, "topology": "supported" if supported else "needs_topology_review",
                     "shared_chunks": len(shared_chunks), "shared_law_scope": len(shared_scope)})
    with open(os.path.join(G, "merge_topology_audit.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(stats, open(os.path.join(G, "topology_audit_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[TOPOLOGY-AUDIT]", stats)


if __name__ == "__main__":
    main()
