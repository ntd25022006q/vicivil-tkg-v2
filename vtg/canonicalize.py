"""canonicalize: S1-S3 pipeline over LightRAG entities.
S1 surface-normalize + group by norm_key
S2 typing (rules) + literal demotion
S3 adjudication tiers: rule-merge -> cosine(embedding matrix) -> LLM (ollama, small model) -> human queue
Everything logged to merge_log.jsonl (rollback = union-find replay without later log entries).
"""
import json
import math
import os
import re
import time
import urllib.request
from collections import defaultdict

from .textops import classify_entity, fold_text, norm_surface


def ollama_same_entity(na, nb, model="qwen3.5:4b", timeout=120):
    """Ask a small local LLM whether two names denote the SAME legal entity.
    Returns (same: bool|None, confidence: float|None)."""
    prompt = (
        "Ban la chuyen gia phap luat Viet Nam. Hai ten sau day co phai la CUNG MOT thuc the "
        "phap luat (cung mot toa an / co quan / van ban / khai niem) khong?\n"
        f'Ten A: "{na}"\nTen B: "{nb}"\n'
        'Tra loi CHI mot dong JSON: {"same": true|false, "confidence": 0.0-1.0}'
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": "json", "think": False,
                       "options": {"temperature": 0, "num_predict": 40}}).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8")).get("response", "")
        same = re.search(r'"same"\s*:\s*(true|false)', out, re.I)
        conf = re.search(r'"confidence"\s*:\s*([\d.]+)', out)
        if not same:
            return None, None
        return same.group(1).lower() == "true", (float(conf.group(1)) if conf else 0.5)
    except Exception:
        return None, None


def load_vdb(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data", obj) if isinstance(obj, dict) else obj
    matrix = obj.get("matrix") if isinstance(obj, dict) else None
    dim = obj.get("embedding_dim") if isinstance(obj, dict) else None
    return data, matrix, dim


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def lexical_sim(a, b):
    """Character-bigram Jaccard — deterministic, dependency-free tier-2 proxy."""
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            return True
        return False


# ---- lexical guards: embeddings do not encode negation/composition — block risky merges
_NEG_PHRASES = [("khong",), ("huy",), ("bai", "bo"), ("hoan",), ("vo", "hieu"), ("to", "hop")]


def _has_phrase(phrase, toks):
    n = len(phrase)
    return any(toks[i:i + n] == list(phrase) for i in range(len(toks) - n + 1))


def merge_blocked(na: str, nb: str) -> bool:
    """True if two names differ by a negation/composition marker or >=3 content tokens."""
    ta = fold_text(na or "").split()
    tb = fold_text(nb or "").split()
    for ph in _NEG_PHRASES:
        if _has_phrase(ph, ta) != _has_phrase(ph, tb):
            return True
    if len(set(ta) ^ set(tb)) >= 3:
        return True
    return False


def run(ws, out_dir, use_llm=False, llm_model="qwen2.5:1.5b",
        cos_auto=0.92, cos_llm=0.80, log=print):
    t0 = time.time()
    src = os.path.join(ws, "hf_release", "index", "vdb_entities.json")
    data, matrix, dim = load_vdb(src)
    log(f"[S1] loaded {len(data)} entities (matrix={matrix is not None}, dim={dim})")
    ent_by_id = {e.get("__id__", f"ent-{i}"): e for i, e in enumerate(data)}

    # ---- S1: norm_key grouping
    groups = defaultdict(list)
    for e in data:
        name = e.get("entity_name", "")
        e["_norm"] = norm_surface(name)
        groups[e["_norm"] if e["_norm"] else f"__empty_{e.get('__id__')}"].append(e)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    log(f"[S1] distinct norm_keys={len(groups)} | multi-variant groups={len(multi)}")

    # ---- S2: typing + literal demotion
    typed, literals = [], []
    for e in data:
        cls, lit, conf, method = classify_entity(e.get("entity_name", ""))
        e["_class"], e["_lit"], e["_conf"], e["_tmethod"] = cls, lit, conf, method
        (literals if lit else typed).append(e)
    log(f"[S2] typed={len(typed)} literal-demotion={len(literals)}")

    # ---- S3: adjudication
    uf = UnionFind()
    merge_log = []

    def record(method, conf, kept_id, merged_id, evidence):
        uf.union(kept_id, merged_id)
        merge_log.append({"method": method, "confidence": round(conf, 4),
                          "kept": kept_id, "merged": merged_id, "evidence": evidence,
                          "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})

    # tier 1: same norm_key (always merge — same surface after normalization)
    for k, members in groups.items():
        if len(members) > 1:
            keep = members[0].get("__id__")
            for m in members[1:]:
                record("tier1-normkey", 1.0, keep, m.get("__id__"), {"norm_key": k,
                       "variants": [m0.get("entity_name") for m0 in members]})

    # tier 2: cosine on embedding matrix (fallback lexical) — blocked by norm-key prefix buckets
    id_to_idx = {e.get("__id__"): i for i, e in enumerate(data)}
    by_class = defaultdict(list)
    for e in typed:
        by_class[e["_class"]].append(e)
    pair_scanned = tier2_merges = 0
    pending = []
    blocked_guard = 0
    MAX_BUCKET = 400          # above this, compare each item only to WINDOW neighbors
    WINDOW = 25
    for cls, members in by_class.items():
        for e in members:
            e["_nsort"] = e.get("_norm", "")
        members.sort(key=lambda e: e["_nsort"])
        buckets = defaultdict(list)
        for e in members:
            buckets[e["_nsort"][:10]].append(e)
        for bkey, bucket in buckets.items():
            n = len(bucket)
            if n > MAX_BUCKET:
                pairs = [(bucket[i], bucket[i + k]) for i in range(n)
                         for k in range(1, WINDOW + 1) if i + k < n]
            else:
                pairs = [(bucket[i], bucket[j]) for i in range(n)
                         for j in range(i + 1, n)]
            for a, b in pairs:
                ra, rb = uf.find(a["__id__"]), uf.find(b["__id__"])
                if ra == rb:
                    continue
                na, nb = a.get("entity_name", ""), b.get("entity_name", "")
                if merge_blocked(na, nb):
                    blocked_guard += 1
                    continue
                lsim = lexical_sim(a["_nsort"], b["_nsort"])
                if lsim < 0.5:
                    continue
                pair_scanned += 1
                sim = None
                if matrix:
                    try:
                        ia, ib = id_to_idx[a["__id__"]], id_to_idx[b["__id__"]]
                        va, vb = matrix[ia], matrix[ib]
                        if va and vb:
                            sim = cosine(va, vb)
                    except Exception:
                        sim = None
                if sim is None:
                    sim = lsim
                if sim >= cos_auto:
                    record("tier2-cosine", sim, a["__id__"], b["__id__"],
                           {"names": [na, nb], "class": cls})
                    tier2_merges += 1
                elif sim >= cos_llm:
                    pending.append((a["__id__"], b["__id__"], sim, na, nb, cls))
    log(f"[S3] pairs scanned={pair_scanned} tier2 auto-merges={tier2_merges} ({time.time()-t0:.1f}s)")

    # tier 3: LLM adjudication for cos_llm <= sim < cos_auto via small local model
    tier3_merges = tier3_rejected = tier3_errors = 0
    log(f"[S3] tier3 candidates={len(pending)} use_llm={use_llm} model={llm_model}")
    if use_llm and pending:
        for (aid, bid, sim, na, nb, cls) in pending[:300]:
            if merge_blocked(na, nb):
                tier3_rejected += 1
                merge_log.append({"method": "tier3-guard-blocked", "confidence": sim,
                                  "kept": aid, "merged": bid, "evidence": {"names": [na, nb]},
                                  "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
                continue
            same, conf = ollama_same_entity(na, nb, model=llm_model)
            if same is None:
                tier3_errors += 1
                merge_log.append({"method": "tier3-llm-error", "confidence": sim,
                                  "kept": aid, "merged": bid, "evidence": {"names": [na, nb]},
                                  "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            elif same and (conf or 0) >= 0.85:
                record("tier3-llm", conf, aid, bid, {"names": [na, nb], "class": cls, "sim": sim})
                tier3_merges += 1
            else:
                tier3_rejected += 1
                merge_log.append({"method": "tier3-llm-rejected", "confidence": conf,
                                  "kept": aid, "merged": bid, "evidence": {"names": [na, nb]},
                                  "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        log(f"[S3] tier3: merged={tier3_merges} rejected={tier3_rejected} errors={tier3_errors}")
    log(f"[S3] guard-blocked pairs={blocked_guard}")

    # ---- output
    os.makedirs(out_dir, exist_ok=True)
    roots = {}
    for e in data:
        r = uf.find(e["__id__"])
        roots.setdefault(r, []).append(e)
    entities_out = []
    for idx, (root, members) in enumerate(sorted(roots.items())):
        names = [m.get("entity_name", "") for m in members]
        # preferred label = most frequent length-weighted (longest name as heuristic canonical form)
        preferred = max(names, key=lambda n: len(n))
        classes = {m["_class"] for m in members}
        cls = members[0]["_class"] if len(classes) == 1 else f"CONFLICT:{sorted(classes)}"
        scopes = sorted({s for m in members for s in (m.get("law_ids") or [])})
        entities_out.append({
            "canonical_id": f"ENT-{idx:06d}",
            "class": cls,
            "preferred_label": preferred,
            "aliases": sorted(set(names)),
            "norm_key": members[0].get("_norm", ""),
            "law_scope": scopes,
            "chunk_ids": sorted({c for m in members for c in (m.get("chunk_ids") or [])})[:200],
            "member_count": len(members),
            "confidence": min(m["_conf"] for m in members),
            "method": members[0]["_tmethod"],
        })
    with open(os.path.join(out_dir, "nodes_entity.jsonl"), "w", encoding="utf-8") as f:
        for e in entities_out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "merge_log.jsonl"), "w", encoding="utf-8") as f:
        for m in merge_log:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    # pending pairs (tier-3 band, unprocessed) → human queue input
    with open(os.path.join(out_dir, "pending_pairs.jsonl"), "w", encoding="utf-8") as f:
        for (aid, bid, sim, na, nb, cls) in pending:
            f.write(json.dumps({"a_id": aid, "b_id": bid, "sim": round(sim, 4),
                                "name_a": na, "name_b": nb, "class": cls},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "literals.jsonl"), "w", encoding="utf-8") as f:
        for e in literals:
            f.write(json.dumps({"name": e.get("entity_name"), "class": "vclo:Literal",
                                "chunk_ids": (e.get("chunk_ids") or [])[:50]}, ensure_ascii=False) + "\n")
    stats = {
        "entities_in": len(data),
        "distinct_norm_keys": len(groups),
        "multi_variant_groups_tier1": len(multi),
        "tier1_merges": sum(1 for m in merge_log if m["method"] == "tier1-normkey"),
        "tier2_merges": tier2_merges,
        "tier3_merges": tier3_merges,
        "tier3_rejected": tier3_rejected,
        "tier3_errors": tier3_errors,
        "guard_blocked_pairs": blocked_guard,
        "canonical_entities_out": len(entities_out),
        "literals_demoted": len(literals),
        "typed_in": len(typed),
        "untyped_out": 0,
        "matrix_used": matrix is not None,
        "llm_model": llm_model if use_llm else None,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "canonicalize_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    log(f"[OUT] canonical entities={len(entities_out)} merges logged={len(merge_log)} stats={stats}")
    return stats
