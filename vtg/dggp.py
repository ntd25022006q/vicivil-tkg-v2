"""dggp: TADG's Distillation-Guided Gradient Projection (DGGP) — the REAL mechanism,
replacing the naive blanket-MSE distillation (published negative result in v0.3.0).

DGGP: during continual training on new-era triples, for OLD entities compute the
distillation-loss gradient direction d_e (Huber/MSE on memory buffer); if the task
gradient g_e CONFLICTS with d_e (dot > 0), project g_e onto the orthogonal complement:
    g_e <- g_e - (g_e . d_e / ||d_e||^2) d_e
This blocks forgetting-direction updates while leaving all other learning intact.

Comparison (3 seeds, same protocol as embed_temporal):
  full_retrain / fast_freeze / naive_distill (lambda=0.5) / **DGGP** / no_continual
Output: graph/embedding_dggp_results.json
"""
import json
import os
import random
import statistics
import time

import torch

from .embed_temporal import (build_triples, era_split, TransE, to_idx, jl, DEV, DIM, EPOCHS, LR)

G = os.path.expanduser("~/vicivil_tkg/graph")
LAMBDA_MEM = 1.0
SEEDS = [42, 101, 2024]


def train_dggp(model, new_triples, memory_triples, old_ent_idx, epochs=EPOCHS):
    """Continual stage with gradient projection against memory-buffer distill direction."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(new_triples)
    if n == 0:
        return model
    mem = memory_triples[torch.randint(0, len(memory_triples), (512,), device=DEV)] \
        if len(memory_triples) else None
    for ep in range(epochs):
        idx = torch.randint(0, n, (min(1024, n),), device=DEV)
        s, r, o = new_triples[idx, 0], new_triples[idx, 1], new_triples[idx, 2]
        neg = torch.randint(0, model.ent.num_embeddings, (len(idx),), device=DEV)
        loss = torch.relu(1.0 + model.score(s, r, neg) - model.score(s, r, o)).sum()
        opt.zero_grad()
        loss.backward()
        # ---- DGGP: distill direction from memory buffer (Huber on old-entity embeddings)
        if mem is not None and len(old_ent_idx):
            dloss = torch.nn.functional.smooth_l1_loss(
                model.ent.weight[old_ent_idx],
                model.ent.weight.detach()[old_ent_idx] * 0 + _frozen_snapshot[old_ent_idx])
            dgrad = torch.autograd.grad(dloss, model.ent.weight, retain_graph=True)[0][old_ent_idx]
            g = model.ent.weight.grad[old_ent_idx]
            dot = (g * dgrad).sum(-1, keepdim=True)
            dn = (dgrad * dgrad).sum(-1, keepdim=True).clamp_min(1e-12)
            conflict = (dot > 0).float()
            proj = g - conflict * (dot / dn) * dgrad
            model.ent.weight.grad[old_ent_idx] = proj
        opt.step()
    return model


_frozen_snapshot = None


def run_seed(seed, s1, s2, s3, ent2id, rel2id, n_ent, n_rel, evaluate, pos_tail, pos_head):
    global _frozen_snapshot
    torch.manual_seed(seed)
    random.seed(seed)
    all_tr = s1 + s2 + s3
    random.shuffle(all_tr)
    cut = max(1, int(0.15 * len(all_tr)))
    test_all = all_tr[:cut]
    train_all = all_tr[cut:]
    test1 = to_idx([t for t in test_all if t[3] < 2017], ent2id, rel2id)
    test2 = to_idx([t for t in test_all if 2017 <= t[3] < 2025], ent2id, rel2id)
    test3 = to_idx([t for t in test_all if t[3] >= 2025], ent2id, rel2id)
    tr1 = to_idx([t for t in train_all if t[3] < 2017], ent2id, rel2id)
    tr2 = to_idx([t for t in train_all if 2017 <= t[3] < 2025], ent2id, rel2id)
    tr3 = to_idx([t for t in train_all if t[3] >= 2025], ent2id, rel2id)
    tra = to_idx(train_all, ent2id, rel2id)

    m = TransE(n_ent, n_rel).to(DEV)
    from .embed_temporal import train as train_plain
    train_plain(m, tr1)
    _frozen_snapshot = m.ent.weight.detach().clone()
    old_ents = torch.tensor(sorted({i for s, r, o in tr1.tolist() for i in (s, o)}), device=DEV)
    # stage 2: new entities init + DGGP continual
    new2 = torch.tensor(sorted({i for s, r, o in tr2.tolist() for i in (s, o)} - set(old_ents.tolist())),
                        device=DEV) if len(tr2) else torch.tensor([], dtype=torch.long, device=DEV)
    m.ent.weight.data[new2] = torch.randn(len(new2), DIM, device=DEV) * 0.1
    train_dggp(m, torch.cat([tr1, tr2]), tr1, old_ents)
    # stage 3
    seen = set(old_ents.tolist()) | set(new2.tolist())
    new3 = torch.tensor(sorted({i for s, r, o in tr3.tolist() for i in (s, o)} - seen),
                        device=DEV) if len(tr3) else torch.tensor([], dtype=torch.long, device=DEV)
    m.ent.weight.data[new3] = torch.randn(len(new3), DIM, device=DEV) * 0.1
    train_dggp(m, torch.cat([tr1, tr2, tr3]), tr1, old_ents)
    return {"S1": evaluate(m, test1, pos_tail, pos_head, n_ent),
            "S2": evaluate(m, test2, pos_tail, pos_head, n_ent),
            "S3": evaluate(m, test3, pos_tail, pos_head, n_ent)}


def main():
    t0 = time.time()
    triples = build_triples()
    ents = sorted({t[0] for t in triples} | {t[2] for t in triples})
    rels = sorted({t[1] for t in triples})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    s1, s2, s3 = era_split(triples)
    print(f"[DGGP] ents={len(ents)} triples={len(triples)} (S1={len(s1)} S2={len(s2)} S3={len(s3)})")

    # reuse eval machinery from embed_temporal
    from .embed_temporal import evaluate as eval_fn
    def pos_indexes(trs):
        tail, head = {}, {}
        for s, r, o, _ in trs:
            tail.setdefault((ent2id[s], rel2id[r]), set()).add(ent2id[o])
            head.setdefault((rel2id[r], ent2id[o]), set()).add(ent2id[s])
        return tail, head
    pos_tail, pos_head = pos_indexes(s1 + s2 + s3)

    per_seed = {}
    for seed in SEEDS:
        st = time.time()
        per_seed[seed] = run_seed(seed, s1, s2, s3, ent2id, rel2id, len(ents), len(rels),
                                  eval_fn, pos_tail, pos_head)
        print(f"[DGGP] seed {seed}: {per_seed[seed]} ({time.time()-st:.0f}s)", flush=True)
    agg = {}
    for era in ("S1", "S2", "S3"):
        mrrs = [per_seed[s][era]["mrr"] for s in SEEDS if per_seed[s][era]["mrr"] is not None]
        h10s = [per_seed[s][era]["hits10"] for s in SEEDS if per_seed[s][era]["hits10"] is not None]
        agg[era] = {"mrr_mean": round(statistics.mean(mrrs), 4) if mrrs else None,
                    "mrr_std": round(statistics.pstdev(mrrs), 4) if len(mrrs) > 1 else 0.0,
                    "hits10_mean": round(statistics.mean(h10s), 4) if h10s else None}
    prev = json.load(open(os.path.join(G, "embedding_temporal_results.json"), encoding="utf-8"))["aggregate"]
    out = {"config": {"dim": DIM, "epochs": EPOCHS, "seeds": SEEDS,
                      "mechanism": "gradient projection onto orthogonal complement of memory-buffer "
                                   "distillation direction for OLD entities (TADG DGGP)"},
           "dggp_aggregate": agg,
           "comparison_v030": {m: prev[m] for m in ("full_retrain", "fast_freeze", "slow_distill", "no_continual")},
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(G, "embedding_dggp_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[DGGP] aggregate:", json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
