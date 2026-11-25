"""embed_temporal: TADG-inspired continual KG embedding on the REAL ViCivil norm graph.
Protocol (mirrors continual KGE literature: successive snapshots, forgetting vs plasticity):
  - Triples: document/article-level edges (SUCCEEDS/CITES/AMENDS/REPLACES/CONSOLIDATES/GUIDES) + inverses
  - Era per edge: year(edge.at) if present else valid_from of the source entity's law
  - S1 = era < 2017, S2 = 2017-2024, S3 = >= 2025
  - Models compared (dim=128, 3 seeds):
      * full-retrain  : train from scratch on S1+S2(+S3)
      * fast-freeze   : copy old embeddings frozen, train only new entities (TADG fast stage)
      * slow-distill  : unfreeze all + lambda * MSE(new, frozen_old) (TADG distillation-guided)
      * no-continual  : train only on S2(+S3) — lower bound for old-era performance
  - Metrics: filtered MRR / Hits@10 on held-out test triples per era (older-era drop = forgetting)
Output: graph/embedding_temporal_results.json
"""
import json
import os
import random
import time

import torch

G = os.path.expanduser("~/vicivil_tkg/graph")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 128
EPOCHS = 250
LR = 0.01
LAMBDA = 0.5
SEEDS = [42, 101, 2024]


def jl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def build_triples():
    edges = jl(os.path.join(G, "edges_norm.jsonl"))
    docs = {n["node_id"]: n for n in jl(os.path.join(G, "nodes_legal.jsonl"))}
    triples = []  # (s, r, o, era:int)
    for e in edges:
        s, r, o, at = e["src"], e["relation"], e["tgt"], e.get("at")
        era = None
        if at:
            era = int(str(at)[:4])
        else:
            src_law = s.split("|")[0] if "|" in s else s
            vf = (docs.get(src_law) or {}).get("valid_from")
            era = int(str(vf)[:4]) if vf else None
        if era is None:
            era = 2026
        triples.append((s, r, o, era))
        triples.append((o, r + "_inv", s, era))
    return triples


def era_split(triples):
    s1 = [t for t in triples if t[3] < 2017]
    s2 = [t for t in triples if 2017 <= t[3] < 2025]
    s3 = [t for t in triples if t[3] >= 2025]
    return s1, s2, s3


class TransE(torch.nn.Module):
    def __init__(self, n_ent, n_rel, dim=DIM):
        super().__init__()
        self.ent = torch.nn.Embedding(n_ent, dim)
        self.rel = torch.nn.Embedding(n_rel, dim)
        torch.nn.init.xavier_uniform_(self.ent.weight)
        torch.nn.init.xavier_uniform_(self.rel.weight)

    def score(self, s, r, o):
        return -torch.norm(self.ent(s) + self.rel(r) - self.ent(o), p=2, dim=-1)


def train(model, triples_idx, epochs=EPOCHS, freeze_ent=None,
          distill_to=None, lam=LAMBDA, log_every=250):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(triples_idx)
    if n == 0:
        return model
    for ep in range(epochs):
        idx = torch.randint(0, n, (min(1024, n),), device=DEV)
        s, r, o = triples_idx[idx, 0], triples_idx[idx, 1], triples_idx[idx, 2]
        neg = torch.randint(0, model.ent.num_embeddings, (len(idx),), device=DEV)
        pos = model.score(s, r, o)
        negd = model.score(s, r, neg)
        loss = torch.relu(1.0 + negd - pos).sum()
        if distill_to is not None and lam > 0:
            d = ((model.ent.weight - distill_to) ** 2).sum(-1)
            loss = loss + lam * d.sum()
        opt.zero_grad()
        loss.backward()
        if freeze_ent is not None:
            # keep frozen rows fixed after backward (fast stage)
            model.ent.weight.grad[freeze_ent] = 0
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, test_idx, pos_tail, pos_head, ent_count, sample=300):
    """Filtered MRR/Hits@10, both directions, batched cdist (B x E) — memory-safe."""
    if len(test_idx) == 0:
        return {"mrr": None, "hits10": None, "n": 0}
    perm = torch.randperm(len(test_idx))[:sample]
    te = test_idx[perm]
    W = model.ent.weight  # (E, D)
    ranks = []
    for mode in ("tail", "head"):
        if mode == "tail":
            q = W[te[:, 0]] + model.rel.weight[te[:, 1]]        # (B, D)
            sc = -torch.cdist(q, W)                             # (B, E)
            golds = te[:, 2]
        else:
            q = W[te[:, 2]] + model.rel.weight[te[:, 1]]
            sc = -torch.cdist(q, W)
            golds = te[:, 0]
        sc = sc.cpu()
        for j in range(len(te)):
            s, r, o = te[j].tolist()
            positives = (pos_tail.get((s, r), set()) if mode == "tail"
                         else pos_head.get((r, o), set()))
            gold = o if mode == "tail" else s
            for p in positives:
                if p != gold:
                    sc[j, p] = -1e9
            rk = int((sc[j] > sc[j, gold]).sum().item()) + 1
            ranks.append(rk)
    mrr = sum(1.0 / rk for rk in ranks) / len(ranks)
    h10 = sum(1.0 for rk in ranks if rk <= 10) / len(ranks)
    return {"mrr": round(mrr, 4), "hits10": round(h10, 4), "n": len(ranks)}


def to_idx(triples, ent2id, rel2id):
    return torch.tensor([[ent2id[s], rel2id[r], ent2id[o]] for s, r, o, _ in triples],
                        dtype=torch.long, device=DEV)


def run_seed(seed, s1, s2, s3, ent2id, rel2id, n_ent, n_rel):
    torch.manual_seed(seed)
    random.seed(seed)
    res = {}
    # position indexes for filtered eval (both directions)
    def pos_indexes(triples):
        tail, head = {}, {}
        for s, r, o, _ in triples:
            tail.setdefault((ent2id[s], rel2id[r]), set()).add(ent2id[o])
            head.setdefault((rel2id[r], ent2id[o]), set()).add(ent2id[s])
        return tail, head

    all_tr = s1 + s2 + s3
    random.shuffle(all_tr)
    cut = max(1, int(0.15 * len(all_tr)))
    test_all = all_tr[:cut]
    train_all = all_tr[cut:]
    test1 = [t for t in test_all if t[3] < 2017]
    test2 = [t for t in test_all if 2017 <= t[3] < 2025]
    test3 = [t for t in test_all if t[3] >= 2025]
    pos_tail, pos_head = pos_indexes(all_tr)

    t1i, t2i, t3i = to_idx(s1, ent2id, rel2id), to_idx(s2, ent2id, rel2id), to_idx(s3, ent2id, rel2id)
    tr1 = to_idx([t for t in train_all if t[3] < 2017], ent2id, rel2id)
    tr2 = to_idx([t for t in train_all if 2017 <= t[3] < 2025], ent2id, rel2id)
    tr3 = to_idx([t for t in train_all if t[3] >= 2025], ent2id, rel2id)
    tra = to_idx(train_all, ent2id, rel2id)

    # --- baseline: full retrain on all eras
    m_full = TransE(n_ent, n_rel).to(DEV)
    train(m_full, tra)
    res["full_retrain"] = {"S1": evaluate(m_full, to_idx(test1, ent2id, rel2id), pos_tail, pos_head, n_ent),
                           "S2": evaluate(m_full, to_idx(test2, ent2id, rel2id), pos_tail, pos_head, n_ent),
                           "S3": evaluate(m_full, to_idx(test3, ent2id, rel2id), pos_tail, pos_head, n_ent)}

    # --- continual: S1 -> +S2 -> +S3 with fast-freeze and slow-distill
    m_fast = TransE(n_ent, n_rel).to(DEV)
    train(m_fast, tr1)
    old_ents = torch.tensor(sorted({i for s, r, o in tr1.tolist() for i in (s, o)}), device=DEV)
    frozen_fast = m_fast.ent.weight.detach().clone()
    new_ents_s2 = torch.tensor(sorted({i for s, r, o in tr2.tolist() for i in (s, o)}
                                      - set(old_ents.tolist())), device=DEV) if len(tr2) else torch.tensor([], dtype=torch.long, device=DEV)
    m_fast.ent.weight.data[new_ents_s2] = torch.randn(len(new_ents_s2), DIM, device=DEV) * 0.1
    train(m_fast, torch.cat([tr1, tr2]), freeze_ent=old_ents)
    frozen2 = m_fast.ent.weight.detach().clone()
    new_ents_s3 = torch.tensor(sorted({i for s, r, o in tr3.tolist() for i in (s, o)}
                                      - set(torch.cat([old_ents, new_ents_s2]).tolist())), device=DEV) if len(tr3) else torch.tensor([], dtype=torch.long, device=DEV)
    m_fast.ent.weight.data[new_ents_s3] = torch.randn(len(new_ents_s3), DIM, device=DEV) * 0.1
    train(m_fast, torch.cat([tr1, tr2, tr3]), freeze_ent=torch.cat([old_ents, new_ents_s2]))
    res["fast_freeze"] = {"S1": evaluate(m_fast, to_idx(test1, ent2id, rel2id), pos_tail, pos_head, n_ent),
                          "S2": evaluate(m_fast, to_idx(test2, ent2id, rel2id), pos_tail, pos_head, n_ent),
                          "S3": evaluate(m_fast, to_idx(test3, ent2id, rel2id), pos_tail, pos_head, n_ent)}

    m_slow = TransE(n_ent, n_rel).to(DEV)
    m_slow.ent.weight.data.copy_(frozen_fast)
    train(m_slow, torch.cat([tr1, tr2]), distill_to=frozen_fast)
    train(m_slow, torch.cat([tr1, tr2, tr3]), distill_to=frozen_fast)
    res["slow_distill"] = {"S1": evaluate(m_slow, to_idx(test1, ent2id, rel2id), pos_tail, pos_head, n_ent),
                           "S2": evaluate(m_slow, to_idx(test2, ent2id, rel2id), pos_tail, pos_head, n_ent),
                           "S3": evaluate(m_slow, to_idx(test3, ent2id, rel2id), pos_tail, pos_head, n_ent)}

    # --- lower bound: no continual (only new eras)
    m_nc = TransE(n_ent, n_rel).to(DEV)
    train(m_nc, torch.cat([tr2, tr3]))
    res["no_continual"] = {"S1": evaluate(m_nc, to_idx(test1, ent2id, rel2id), pos_tail, pos_head, n_ent),
                           "S2": evaluate(m_nc, to_idx(test2, ent2id, rel2id), pos_tail, pos_head, n_ent),
                           "S3": evaluate(m_nc, to_idx(test3, ent2id, rel2id), pos_tail, pos_head, n_ent)}
    return res


def main():
    t0 = time.time()
    triples = build_triples()
    ents = sorted({t[0] for t in triples} | {t[2] for t in triples})
    rels = sorted({t[1] for t in triples})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    s1, s2, s3 = era_split(triples)
    print(f"[EMB] ents={len(ents)} rels={len(rels)} triples={len(triples)} "
          f"(S1={len(s1)} S2={len(s2)} S3={len(s3)}) dev={DEV}")
    per_seed = {}
    for seed in SEEDS:
        st = time.time()
        per_seed[seed] = run_seed(seed, s1, s2, s3, ent2id, rel2id, len(ents), len(rels))
        print(f"[EMB] seed {seed} done in {time.time()-st:.0f}s")
    # aggregate mean±std
    import statistics
    agg = {}
    for model in ("full_retrain", "fast_freeze", "slow_distill", "no_continual"):
        agg[model] = {}
        for era in ("S1", "S2", "S3"):
            mrrs = [per_seed[s][model][era]["mrr"] for s in SEEDS
                    if per_seed[s][model][era]["mrr"] is not None]
            h10s = [per_seed[s][model][era]["hits10"] for s in SEEDS
                    if per_seed[s][model][era]["hits10"] is not None]
            agg[model][era] = {
                "mrr_mean": round(statistics.mean(mrrs), 4) if mrrs else None,
                "mrr_std": round(statistics.pstdev(mrrs), 4) if len(mrrs) > 1 else 0.0,
                "hits10_mean": round(statistics.mean(h10s), 4) if h10s else None,
                "n_eval_per_seed": per_seed[SEEDS[0]][model][era]["n"],
            }
    out = {"config": {"dim": DIM, "epochs": EPOCHS, "seeds": SEEDS, "lambda_distill": LAMBDA,
                      "snapshot_rule": "era<2017 / 2017-2024 / >=2025",
                      "graph": "edges_norm (SUCCEEDS/CITES/AMENDS/REPLACES/CONSOLIDATES/GUIDES + inverses)"},
           "per_seed": {str(k): v for k, v in per_seed.items()},
           "aggregate": agg,
           "note": "S1 MRR drop from full_retrain to fast_freeze = forgetting; "
                   "slow_distill (TADG distillation-guided) should recover part of it",
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(G, "embedding_temporal_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[EMB] aggregate:")
    print(json.dumps(agg, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
