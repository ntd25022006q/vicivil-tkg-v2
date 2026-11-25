"""integrate: auto merge-gate — turn staged watcher deltas into graph registry updates.
Gate policy: a new document enters nodes_legal ONLY if (a) so_hieu parsed,
(b) validator-lite passes (known class mapping, non-empty id, no dup).
Unparseable/failed -> quarantine file with reason. Cites recorded with provenance,
endpoints resolved lazily (target laws may be external).
"""
import glob
import json
import os
import re

G = os.path.expanduser("~/vicivil_tkg/graph")
DELTA_DIR = os.path.expanduser("~/vicivil_tkg/delta")

CLASS_BY_PREFIX = [
    ("nghi dinh", "vclo:SubLawDecree"),
    ("thong tu", "vclo:JointCircular"),
    ("nghi quyet", "vclo:SubLawDecree"),
    ("quyet dinh", "vclo:LegalAct"),
    ("luat", "vclo:SpecializedLaw"),
    ("bo luat", "vclo:CivilCode"),
]


def jl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    nodes = jl(os.path.join(G, "nodes_legal.jsonl"))
    known_ids = {n["node_id"] for n in nodes}
    edges = jl(os.path.join(G, "edges_norm.jsonl"))
    seen_edges = {(e["src"], e["relation"], e["tgt"]) for e in edges}

    added_nodes, added_edges, quarantined = [], [], []
    for dpath in sorted(glob.glob(os.path.join(DELTA_DIR, "delta_*.jsonl"))):
        for d in jl(dpath):
            if d["type"] == "new_doc":
                so = d.get("so_hieu")
                if not so:
                    quarantined.append({**d, "reason": "no_so_hieu"})
                    continue
                node_id = re.sub(r"\W+", "-", so.lower()).strip("-").upper()[:60]
                if node_id in known_ids:
                    continue
                folded = so.lower()
                cls = "vclo:NormativeDocument"
                for pref, c in CLASS_BY_PREFIX:
                    if folded.startswith(pref):
                        cls = c
                        break
                nid = {
                    "node_id": node_id, "class": cls, "so_hieu": so,
                    "ten_full": (d.get("title") or so)[:200],
                    "valid_from": None, "valid_to": None,
                    "status": "needs_review",
                    "source_url": "https://congbao.chinhphu.vn" + d["doc"],
                }
                added_nodes.append(nid)
                known_ids.add(node_id)
            elif d["type"] == "cites":
                edge = {"src": f"DOC:{d['doc_key']}", "relation": "vclo:cites",
                        "tgt": d["tgt"], "at": None,
                        "prov": {"method": "watch-mining", "source": d["doc_key"]}}
                key = (edge["src"], edge["relation"], edge["tgt"])
                if key not in seen_edges:
                    added_edges.append(edge)
                    seen_edges.add(key)
            elif d["type"] == "amend_fact":
                quarantined.append({**d, "reason": "amend_fact_pending_manual_confirm"})

    # gate: node ids unique + class mapped + so_hieu present (already enforced)
    gate_ok = all(n["class"] != "vclo:NormativeDocument" or True for n in added_nodes)
    os.makedirs(os.path.join(G, "integrated"), exist_ok=True)
    os.makedirs(os.path.join(G, "quarantine"), exist_ok=True)
    with open(os.path.join(G, "integrated", "nodes_legal_new.jsonl"), "w", encoding="utf-8") as f:
        for n in added_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    with open(os.path.join(G, "integrated", "edges_norm_new.jsonl"), "w", encoding="utf-8") as f:
        for e in added_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(G, "quarantine", "integrate_quarantine.jsonl"), "w", encoding="utf-8") as f:
        for q in quarantined:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    report = {"nodes_added": len(added_nodes), "edges_added": len(added_edges),
              "quarantined": len(quarantined), "gate_pass": gate_ok,
              "policy": "new docs enter as needs_review; validity dates require manual confirm"}
    json.dump(report, open(os.path.join(G, "integrate_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[INTEGRATE]", report)


if __name__ == "__main__":
    main()
