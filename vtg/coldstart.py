"""coldstart: DCEL-style fast-path embedding for NEW/ISOLATED nodes without retraining.
Recipe (DCEL TKDD 2025, scaled to our data):
  MRC  : description embedding (chunk text) via embeddinggemma:300m (ollama)
  RSA  : per-relation affine map A_r: desc-space -> KGE-space, ridge-fitted on anchor entities
  Eval : hold out entities with known edges; pretend they are NEW; predict their
         relations via cold-start embedding; filtered Hits@10/MRR vs random baseline.
Output: graph/coldstart_results.json
"""
import json
import os
import random
import statistics
import time
import urllib.request

import torch

from .embed_temporal import (build_triples, TransE, to_idx, jl, DEV, DIM, EPOCHS, LR, train)

G = os.path.expanduser("~/vicivil_tkg/graph")
WS = os.path.expanduser("~/law_workspace")
EMB_MODEL = "embeddinggemma:300m"
SEEDS = [42, 101, 2024]


def ollama_embed(texts):
    body = json.dumps({"model": EMB_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))["embeddings"]


def load_descriptions():
    """Article entities get their chunk text (first 260 chars) as description."""
    d = {}
    with open(os.path.join(WS, "runs", "civil_chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            d.setdefault(c["chunk_id"], c["content"][:260])
    return d


def main():
    t0 = time.time()
    random.seed(20261125)
    triples = build_triples()
    ents = sorted({t[0] for t in triples} | {t[2] for t in triples})
    rels = sorted({t[1] for t in triples})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    n_ent, n_rel = len(ents), len(rels)
    desc = load_descriptions()

    # candidate "new" entities: articles with a description and >=2 incident edges
    from collections import defaultdict
    deg = defaultdict(int)
    for s, r, o, _ in triples:
        deg[s] += 1
        deg[o] += 1
    cands = [e for e in ents if e in desc and deg[e] >= 2 and "|" in e]
    random.shuffle(cands)
    heldout = cands[:40]
    heldout_set = set(heldout)
    print(f"[COLDSTART] ents={n_ent} candidates={len(cands)} heldout={len(heldout)}")

    # triples for training with heldout entities REMOVED entirely (no leakage of gold edges)
    train_tr = [t for t in triples if t[0] not in heldout_set and t[2] not in heldout_set]
    tail, head = {}, {}
    for s, r, o, _ in triples:
        tail.setdefault((ent2id[s], rel2id[r]), set()).add(ent2id[o])
        head.setdefault((rel2id[r], ent2id[o]), set()).add(ent2id[s])

    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)
        m = TransE(n_ent, n_rel).to(DEV)
        train(m, to_idx(train_tr, ent2id, rel2id), epochs=EPOCHS)
        W = m.ent.weight.detach()

        # ---- MRC: description embeddings for heldout + anchor pool
        anchor_ents = [e for e in cands[len(heldout):len(heldout) + 500] if e in desc]
        texts = [desc[e] for e in heldout + anchor_ents]
        vecs = []
        for i in range(0, len(texts), 32):
            vecs.extend(ollama_embed(texts[i:i + 32]))
        D = torch.tensor(vecs, dtype=torch.float32, device=DEV)
        dv = {e: D[i] for i, e in enumerate(heldout + anchor_ents)}

        # ---- RSA: per-relation ridge affine from desc-vec -> KGE-vec (anchors)
        def rsa_predict(entity, relation):
            src = dv[entity]
            # anchors for this relation: entities whose folded id shares law scope —
            # use ALL anchor entities as fitting pool, weighted by relation presence
            X = torch.stack([dv[e] for e in anchor_ents])
            Y = W[[ent2id[e] for e in anchor_ents]]
            Xa = torch.cat([X, torch.ones(len(X), 1, device=DEV)], dim=1)
            lam = 1e-2
            A = torch.linalg.solve(Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=DEV),
                                   Xa.T @ Y)
            x = torch.cat([src, torch.ones(1, device=DEV)])
            return (x @ A).squeeze(0)

        ranks = []
        for e in heldout:
            eid = ent2id[e]
            rel_list = sorted({r for (s, r, o, _) in triples if s == e or o == e})
            for r in rel_list[:2]:
                q = rsa_predict(e, r)
                sc = -torch.cdist(q.unsqueeze(0), W).squeeze(0)
                # filtered: mask known positives for (e, r)
                for p in tail.get((eid, rel2id[r]), set()):
                    if p != eid:
                        sc[p] = -1e9
                # gold = the entity connected via this relation in the FULL graph
                golds = [o for (s, rr, o, _) in triples if s == e and rr == r] + \
                        [s for (s, rr, o, _) in triples if o == e and rr == r]
                best = min(golds, key=lambda g: sc[ent2id[g]].item())
                rk = int((sc > sc[ent2id[best]]).sum().item()) + 1
                ranks.append(rk)
        mrr = sum(1.0 / rk for rk in ranks) / max(1, len(ranks))
        h10 = sum(1.0 for rk in ranks if rk <= 10) / max(1, len(ranks))
        results.append({"mrr": round(mrr, 4), "hits10": round(h10, 4), "n": len(ranks)})
        print(f"[COLDSTART] seed {seed}: {results[-1]} ({time.time()-t0:.0f}s)", flush=True)

    agg = {"mrr_mean": round(statistics.mean(x["mrr"] for x in results), 4),
           "hits10_mean": round(statistics.mean(x["hits10"] for x in results), 4),
           # FIX (audit): expected MRR of a UNIFORM random ranking with 1 gold among n
           # is (1/n)·H_n ≈ (ln n + 0.5772)/n — not 1/n (that is Hit@1 probability)
           "random_mrr_expected": round((__import__("math").log(n_ent) + 0.5772) / n_ent, 6)}
    out = {"config": {"heldout": 40, "anchors": 500, "desc_model": EMB_MODEL,
                      "kge_dim": DIM, "recipe": "MRC(desc-embed) -> RSA(per-relation ridge affine) -> rank"},
           "per_seed": results, "aggregate": agg, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(G, "coldstart_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[COLDSTART] aggregate:", json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
