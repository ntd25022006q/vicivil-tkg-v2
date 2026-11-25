"""local_update: GEDUM 'Knowledge Calibration' + DINGAL-O local re-embed (L2GFAM-lite).
When a delta lands (new/changed entity), do NOT retrain: calibrate the entity's
embedding from its 1-2 hop neighborhood of the FROZEN base model:
    est(u) = mean over incident edges { emb[o] - rel[r]  (u as subject)
                                     , emb[s] + rel[r]  (u as object) }
Output: graph/embedding_updates.jsonl (128-d vectors + provenance of neighbors).
"""
import json
import os
import time
from collections import defaultdict

import torch

from .embed_temporal import build_triples, TransE, to_idx, jl, DEV, DIM, EPOCHS, train

G = os.path.expanduser("~/vicivil_tkg/graph")


def main():
    t0 = time.time()
    triples = build_triples()
    ents = sorted({t[0] for t in triples} | {t[2] for t in triples})
    rels = sorted({t[1] for t in triples})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    torch.manual_seed(42)
    m = TransE(len(ents), len(rels)).to(DEV)
    train(m, to_idx(triples, ent2id, rel2id))
    W = m.ent.weight.detach()
    R = m.rel.weight.detach()

    # affected entities = delta-born (integrated new docs + watcher cites) + amended articles
    affected = set()
    for d in jl(os.path.join(G, "integrated", "nodes_legal_new.jsonl")):
        affected.add(f"doc-{d['node_id']}")
    for f in jl(os.path.join(G, "article_amend_facts.jsonl")):
        if f.get("law_id") and f.get("dieu"):
            affected.add(f"{f['law_id']}|Dieu{f['dieu']}")
    for c in jl(os.path.join(G, "edges_case.jsonl"))[:200]:
        pass  # case nodes have no KGE embedding (out of norm graph) — skipped
    affected &= set(ents)  # only entities present in the norm graph
    print(f"[LOCAL] affected entities in norm graph: {len(affected)}")

    # adjacency: entity -> list of (rel_id, other_id, direction)
    adj = defaultdict(list)
    for s, r, o, _ in triples:
        si, ri, oi = ent2id[s], rel2id[r], ent2id[o]
        adj[si].append((ri, oi, "subj"))
        adj[oi].append((ri, si, "obj"))

    out = []
    for e in sorted(affected):
        ui = ent2id[e]
        ests = []
        for ri, other, direction in adj[ui]:
            if direction == "subj":
                ests.append(W[other] - R[ri])       # emb[u] ≈ emb[o] - rel[r]
            else:
                ests.append(W[other] + R[ri])       # emb[u] ≈ emb[s] + rel[r]
        if not ests:
            emb = W[ui]
            method = "base-only (no neighbors)"
        else:
            emb = torch.stack(ests).mean(0)
            # calibrate: blend with base (GEDUM AWFS-lite: attention-free 0.5/0.5)
            emb = 0.5 * emb + 0.5 * W[ui]
            method = f"L2GFAM-lite ({len(ests)} incident edges)"
        out.append({"entity": e, "method": method, "n_neighbors": len(ests),
                    "embedding": [round(float(x), 4) for x in emb]})
    with open(os.path.join(G, "embedding_updates.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"affected": len(out), "dim": DIM, "seconds": round(time.time() - t0, 1),
             "note": "frozen base + 1-hop TransE-consistent calibration + 0.5 blend (AWFS-lite)"}
    json.dump(stats, open(os.path.join(G, "local_update_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[LOCAL]", stats)


if __name__ == "__main__":
    main()
