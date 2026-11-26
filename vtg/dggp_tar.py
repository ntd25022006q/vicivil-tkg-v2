"""dggp_tar: TADG TAR (topology-aware selection) combined with DGGP.
TAR idea: hub entities (high adjacency degree — Điều nguyên tắc, định nghĩa, văn bản
trung tâm) get STRONG protection (always project gradients against forgetting);
leaf entities (low degree) are allowed to drift (chúng thường là điều chi tiết hay sửa).
Implementation: projection applied only when degree >= median degree of old entities.
Output: graph/embedding_dggp_tar_results.json
"""
import json
import os
import random
import statistics
import time

import torch

from .embed_temporal import (build_triples, era_split, TransE, to_idx, evaluate,
                             train, DEV, DIM, EPOCHS, LR)
from .dggp import _frozen_snapshot, run_seed  # noqa: F401 (reuse protocol pieces)

G = os.path.expanduser("~/vicivil_tkg/graph")
SEEDS = [42, 101, 2024]


def train_dggp_tar(model, new_triples, memory_triples, old_ent_idx, protect_mask, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(new_triples)
    if n == 0:
        return model
    for ep in range(epochs):
        idx = torch.randint(0, n, (min(1024, n),), device=DEV)
        s, r, o = new_triples[idx, 0], new_triples[idx, 1], new_triples[idx, 2]
        neg = torch.randint(0, model.ent.num_embeddings, (len(idx),), device=DEV)
        loss = torch.relu(1.0 + model.score(s, r, neg) - model.score(s, r, o)).sum()
        opt.zero_grad()
        loss.backward()
        # memory distill direction (re-sampled every epoch — TADG protocol)
        import vtg.dggp as _D
        frozen_ref = _D._frozen_snapshot
        dloss = torch.nn.functional.smooth_l1_loss(
            model.ent.weight[old_ent_idx], frozen_ref[old_ent_idx])
        dgrad = torch.autograd.grad(dloss, model.ent.weight, retain_graph=True)[0][old_ent_idx]
        g = model.ent.weight.grad[old_ent_idx]
        dot = (g * dgrad).sum(-1, keepdim=True)
        dn = (dgrad * dgrad).sum(-1, keepdim=True).clamp_min(1e-12)
        conflict = ((dot > 0).float() * protect_mask.unsqueeze(1))
        model.ent.weight.grad[old_ent_idx] = g - conflict * (dot / dn) * dgrad
        opt.step()
    return model


def main():
    from .dggp import _frozen_snapshot as _fs  # noqa
    t0 = time.time()
    triples = build_triples()
    ents = sorted({t[0] for t in triples} | {t[2] for t in triples})
    rels = sorted({t[1] for t in triples})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    s1, s2, s3 = era_split(triples)

    pos_tail, pos_head = {}, {}
    for s, r, o, _ in triples:
        pos_tail.setdefault((ent2id[s], rel2id[r]), set()).add(ent2id[o])
        pos_head.setdefault((rel2id[r], ent2id[o]), set()).add(ent2id[s])

    per_seed = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)
        # FIX (audit): split base edges first — inverses stay on the same side
        base = [t for t in (s1 + s2 + s3) if not t[1].endswith("_inv")]
        random.shuffle(base)
        cut = max(1, int(0.15 * len(base)))
        tb, rb = base[:cut], base[cut:]

        def _inv(ts):
            return [(o, r[:-4] if r.endswith("_inv") else r + "_inv", s, e) for s, r, o, e in ts]

        test_all = tb + _inv(tb)
        train_all = rb + _inv(rb)
        tr1 = to_idx([t for t in train_all if t[3] < 2017], ent2id, rel2id)
        tr2 = to_idx([t for t in train_all if 2017 <= t[3] < 2025], ent2id, rel2id)
        tr3 = to_idx([t for t in train_all if t[3] >= 2025], ent2id, rel2id)
        test1 = to_idx([t for t in test_all if t[3] < 2017], ent2id, rel2id)
        test2 = to_idx([t for t in test_all if 2017 <= t[3] < 2025], ent2id, rel2id)
        test3 = to_idx([t for t in test_all if t[3] >= 2025], ent2id, rel2id)

        m = TransE(len(ents), len(rels)).to(DEV)
        train(m, tr1)
        import vtg.dggp as D
        D._frozen_snapshot = m.ent.weight.detach().clone()
        frozen = m.ent.weight.detach().clone()
        old_ents = torch.tensor(sorted({i for s, r, o in tr1.tolist() for i in (s, o)}), device=DEV)
        # TAR: degree-based protection mask (median split)
        deg = torch.zeros(len(ents), device=DEV)
        for s, r, o in tr1.tolist():
            deg[s] += 1
            deg[o] += 1
        med = deg[old_ents].median()
        protect_mask = (deg[old_ents] >= med).float()
        new2 = torch.tensor(sorted({i for s, r, o in tr2.tolist() for i in (s, o)} - set(old_ents.tolist())),
                            device=DEV) if len(tr2) else torch.tensor([], dtype=torch.long, device=DEV)
        m.ent.weight.data[new2] = torch.randn(len(new2), DIM, device=DEV) * 0.1
        train_dggp_tar(m, torch.cat([tr1, tr2]), tr1, old_ents, protect_mask)
        seen = set(old_ents.tolist()) | set(new2.tolist())
        new3 = torch.tensor(sorted({i for s, r, o in tr3.tolist() for i in (s, o)} - seen),
                            device=DEV) if len(tr3) else torch.tensor([], dtype=torch.long, device=DEV)
        m.ent.weight.data[new3] = torch.randn(len(new3), DIM, device=DEV) * 0.1
        train_dggp_tar(m, torch.cat([tr1, tr2, tr3]), tr1, old_ents, protect_mask)
        res = {"S1": evaluate(m, test1, pos_tail, pos_head, len(ents)),
               "S2": evaluate(m, test2, pos_tail, pos_head, len(ents)),
               "S3": evaluate(m, test3, pos_tail, pos_head, len(ents))}
        per_seed[seed] = res
        print(f"[TAR] seed {seed}: {res} ({time.time()-t0:.0f}s)", flush=True)

    agg = {}
    for era in ("S1", "S2", "S3"):
        mrrs = [per_seed[s][era]["mrr"] for s in SEEDS]
        agg[era] = {"mrr_mean": round(statistics.mean(mrrs), 4),
                    "mrr_std": round(statistics.pstdev(mrrs), 4),
                    "hits10_mean": round(statistics.mean(per_seed[s][era]["hits10"] for s in SEEDS), 4)}
    prev = json.load(open(os.path.join(G, "embedding_dggp_results.json"), encoding="utf-8"))
    out = {"config": {"mechanism": "TAR (degree>=median → project) + DGGP + memory resample/epoch",
                      "seeds": SEEDS, "dim": DIM},
           "dggp_tar_aggregate": agg,
           "reference": {"dggp": prev["dggp_aggregate"],
                         "full_retrain": prev["comparison_v030"]["full_retrain"],
                         "fast_freeze": prev["comparison_v030"]["fast_freeze"]},
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(G, "embedding_dggp_tar_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[TAR] aggregate:", json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
