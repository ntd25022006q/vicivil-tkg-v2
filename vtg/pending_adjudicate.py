"""pending_adjudicate: run tier-3 LLM adjudication over ALL remaining pending pairs.
guard-first (negation/token-diff) -> ollama qwen3.5:4b -> merge decisions logged.
Output: graph/pending_adjudicated.jsonl + stats. Registry NOT mutated (merge decisions
are appended to merge_log-style file; integration into canonical entities happens in
the next build run with --resume-queue).
"""
import json
import os
import time

from .canonicalize import ollama_same_entity, merge_blocked

G = os.path.expanduser("~/vicivil_tkg/graph")


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main(cap=1200):
    pending = jl(os.path.join(G, "pending_pairs.jsonl"))
    already = {tuple((m.get("evidence") or {}).get("names", ["", ""])) for m in jl(os.path.join(G, "merge_log.jsonl"))
               if m["method"].startswith("tier3")}
    todo = [p for p in pending if (p["name_a"], p["name_b"]) not in already][:cap]
    print(f"[ADJUDICATE] pending={len(pending)} todo={len(todo)}", flush=True)
    stats = {"merged": 0, "rejected": 0, "guard_blocked": 0, "errors": 0}
    t0 = time.time()
    out = []
    for i, p in enumerate(todo):
        na, nb = p["name_a"], p["name_b"]
        if merge_blocked(na, nb):
            stats["guard_blocked"] += 1
            out.append({**p, "decision": "guard_blocked"})
            continue
        same, conf = ollama_same_entity(na, nb, model="qwen3.5:4b")
        if same is None:
            stats["errors"] += 1
            out.append({**p, "decision": "error"})
        elif same and (conf or 0) >= 0.85:
            stats["merged"] += 1
            out.append({**p, "decision": "merge", "llm_confidence": conf})
        else:
            stats["rejected"] += 1
            out.append({**p, "decision": "rejected", "llm_confidence": conf})
        if (i + 1) % 100 == 0:
            print(f"[ADJUDICATE] {i+1}/{len(todo)} {stats} ({time.time()-t0:.0f}s)", flush=True)
    if not todo:
        print("[ADJUDICATE] nothing to do")
    with open(os.path.join(G, "pending_adjudicated.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["seconds"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(G, "pending_adjudicate_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[ADJUDICATE] FINAL", stats)


if __name__ == "__main__":
    main()
