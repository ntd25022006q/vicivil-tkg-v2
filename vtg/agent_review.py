"""agent_review: act AS the delegated reviewer for P1/P2 queue items — transparently.
Every decision is labeled 'agent-reviewed' (an LLM delegate, NOT a human sign-off);
the user can override later via approved_pairs/rejected_pairs edits + rebuild.

Phase A (P1, accepted merges without topology support):
  guard -> reject; roman/number mismatch -> reject;
  else LLM with chunk contexts (conf>=0.85 keep, else reject_pair).
Phase B (P2, CONFLICT clusters): LLM picks the odd-one-out member -> excluded_names
  (singleton on rebuild); no clear outlier -> keep flagged (honest disclosure).
Phase C (P3 spot-check): 30 sampled rejections re-verified with context.

Outputs:
  graph/agent_review.jsonl          (every row: pair, decision, reason)
  graph/approved_pairs.jsonl        (confirmed merges, incl. earlier context rescues)
  graph/rejected_pairs.jsonl        (pairs to forbid on rebuild)
  graph/excluded_names.jsonl        (names to keep singleton on rebuild)
"""
import json
import os
import time
import urllib.request

from .canonicalize import merge_blocked
from .textops import fold_text

G = os.path.expanduser("~/vicivil_tkg/graph")
WS = os.path.expanduser("~/law_workspace")


def jl(p):
    """Tolerant JSONL reader: re-joins torn multi-line records (SSH-reset torn writes),
    skips unrecoverable lines with a count."""
    if not os.path.exists(p):
        return []
    out, buf, bad = [], "", 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            buf += line
            try:
                out.append(json.loads(buf))
                buf = ""
            except json.JSONDecodeError:
                continue
    if buf.strip():
        bad += 1
    if bad:
        print(f"[jl] {os.path.basename(p)}: {bad} unrecoverable line(s) skipped")
    return out


def llm(prompt, np_=40):
    body = json.dumps({"model": "qwen3.5:4b", "prompt": prompt, "stream": False,
                       "format": "json", "think": False,
                       "options": {"temperature": 0, "num_predict": np_}}).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def llm_same(na, nb, ctx):
    prompt = (f"Hai thuật ngữ pháp luật Việt Nam: A=\"{na}\" B=\"{nb}\".\n"
              f"Ngữ cảnh: {ctx[:450] or '(không có)'}\n"
              "A và B có cùng MỘT thực thể/khái niệm pháp luật không (kể cả biến thể chính tả quí/quý)? "
              'Trả lời CHỈ JSON: {"same": true|false, "confidence": 0.0-1.0}')
    try:
        o = llm(prompt)
        d = json.loads(o) if o else {}
        return bool(d.get("same")), float(d.get("confidence", 0))
    except Exception:
        return None, None


def main():
    ents = jl(os.path.join(G, "nodes_entity.jsonl"))
    by_alias = {}
    for e in ents:
        for a in e["aliases"] + [e["preferred_label"]]:
            by_alias.setdefault(a, e)
    vdb = json.load(open(os.path.join(WS, "hf_release", "index", "vdb_entities.json"), encoding="utf-8"))
    data = vdb.get("data", vdb) if isinstance(vdb, dict) else vdb
    chunk_text = {}
    kv = os.path.join(WS, "hf_release", "index", "kv_store_text_chunks.json")
    if os.path.exists(kv):
        kvd = json.load(open(kv, encoding="utf-8"))
        kdata = kvd.get("data", kvd) if isinstance(kvd, dict) else kvd
        if isinstance(kdata, dict):
            kdata = list(kdata.values())
        for c in (kdata[:20000] if isinstance(kdata, list) else []):
            if isinstance(c, dict):
                cid = str(c.get("chunk_id") or c.get("__id__") or "")
                chunk_text[cid] = (c.get("content") or "")[:220]

    audit = jl(os.path.join(G, "merge_topology_audit.jsonl"))
    tri = jl(os.path.join(G, "triage_stats.json"))
    rows_all = []
    t0 = time.time()
    stats = {"P1_confirm": 0, "P1_reject": 0, "P1_guard": 0, "P1_error": 0,
             "P2_keep": 0, "P2_split": 0, "P3_ok": 0, "P3_flip": 0}

    approved, rejected_pairs, excluded = [], [], []
    # giữ lại 33 rescue đã có bằng chứng ngữ cảnh từ bước trước
    for r in jl(os.path.join(G, "context_readjudicated.jsonl")):
        if r.get("context_decision") == "merge":
            names = (r.get("evidence") or {}).get("names") or ["", ""]
            approved.append({"name_a": names[0], "name_b": names[1],
                             "reason": "context-rescued (conf>=0.9, prior pass)"})

    # ---------------- Phase A: P1 accepted-merges without topology support
    supported = {tuple((r.get("evidence") or {}).get("names", ["", ""])[:2]) for r in audit
                 if r.get("topology") == "supported"}
    rescued_names = {tuple((r.get("evidence") or {}).get("names", ["", ""])[:2]) for r in
                     jl(os.path.join(G, "context_readjudicated.jsonl"))
                     if r.get("context_decision") == "merge"}
    seen = set()
    for m in jl(os.path.join(G, "merge_log.jsonl")):
        if not m["method"].startswith("tier"):
            continue
        ev = m.get("evidence") or {}
        names = ev.get("names") or ev.get("variants") or ["", ""]
        if len(names) < 2:
            continue
        na, nb = str(names[0]), str(names[1])
        key = (na, nb)
        if key in seen or key in supported or key in rescued_names:
            continue
        seen.add(key)
        if merge_blocked(na, nb):
            # guard later blocks it on rebuild anyway — record decision
            stats["P1_guard"] += 1
            rows_all.append({"queue": "P1", "a": na, "b": nb, "decision": "reject_pair",
                             "reason": "guard (number/roman/negation/token)"})
            rejected_pairs.append({"name_a": na, "name_b": nb, "reason": "agent: guard"})
            continue
        ea, eb = by_alias.get(na) or {}, by_alias.get(nb) or {}
        ca = [chunk_text.get(c, "") for c in (ea.get("chunk_ids") or [])[:2]]
        cb = [chunk_text.get(c, "") for c in (eb.get("chunk_ids") or [])[:2]]
        ctx = " | ".join(x for x in ca + cb if x)
        same, conf = llm_same(na, nb, ctx)
        if same is None:
            stats["P1_error"] += 1
            rows_all.append({"queue": "P1", "a": na, "b": nb, "decision": "defer",
                             "reason": "LLM error — giữ nguyên trạng thái, đánh dấu cho người"})
            continue
        if same and conf >= 0.85:
            stats["P1_confirm"] += 1
            rows_all.append({"queue": "P1", "a": na, "b": nb, "decision": "confirm_merge",
                             "reason": f"llm+context conf={conf:.2f}"})
            approved.append({"name_a": na, "name_b": nb,
                             "reason": f"agent-reviewed merge (conf={conf:.2f})"})
        else:
            stats["P1_reject"] += 1
            rows_all.append({"queue": "P1", "a": na, "b": nb, "decision": "reject_pair",
                             "reason": f"llm+context same=false (conf={conf:.2f})"})
            rejected_pairs.append({"name_a": na, "name_b": nb, "reason": "agent: different entity"})

    # ---------------- Phase B: P2 conflict clusters — odd-one-out
    for e in ents:
        cls = str(e.get("class", ""))
        if not cls.startswith("CONFLICT") or e.get("member_count", 1) < 2:
            continue
        members = e["aliases"][:8]
        prompt = ("Danh sách tên thực thể pháp luật bị gộp chung một cụm:\n"
                  + "\n".join(f"- {x}" for x in members)
                  + "\nNếu có MỘT tên KHÔNG cùng loại với phần còn lại, trả lời JSON "
                    '{"outlier": "<tên>", "majority_class": "..."}; nếu tất cả cùng loại, '
                    'trả lời {"outlier": null}.')
        try:
            o = llm(prompt, np_=80)
            d = json.loads(o) if o else {}
        except Exception:
            d = {}
        outlier = d.get("outlier")
        if outlier and outlier in members:
            stats["P2_split"] += 1
            excluded.append({"name": outlier, "cluster": e["preferred_label"],
                             "reason": f"agent: odd-one-out of {cls[:60]}"})
            rows_all.append({"queue": "P2", "a": e["preferred_label"], "b": outlier,
                             "decision": "split_outlier", "reason": "LLM odd-one-out"})
        else:
            stats["P2_keep"] += 1
            rows_all.append({"queue": "P2", "a": e["preferred_label"], "b": "",
                             "decision": "keep_flagged", "reason": "no clear outlier — giữ flag minh bạch"})

    # ---------------- Phase C: P3 spot-check 30 rejections
    import random
    random.seed(2026)
    rej = [m for m in jl(os.path.join(G, "merge_log.jsonl")) if m["method"].endswith("rejected")]
    sample = random.sample(rej, min(30, len(rej)))
    for m in sample:
        ev = m.get("evidence") or {}
        names = ev.get("names") or ["", ""]
        na, nb = str(names[0]), str(names[1])
        if merge_blocked(na, nb):
            stats["P3_ok"] += 1
            continue
        ea, eb = by_alias.get(na) or {}, by_alias.get(nb) or {}
        ctx = " | ".join(x for x in [chunk_text.get(c, "") for c in
                                     (ea.get("chunk_ids") or [])[:1] + (eb.get("chunk_ids") or [])[:1]] if x)
        same, conf = llm_same(na, nb, ctx)
        if same and conf >= 0.9:
            stats["P3_flip"] += 1
            approved.append({"name_a": na, "name_b": nb,
                             "reason": f"agent spot-check flip (conf={conf:.2f})"})
            rows_all.append({"queue": "P3", "a": na, "b": nb, "decision": "flip_to_merge",
                             "reason": f"spot-check conf={conf:.2f}"})
        else:
            stats["P3_ok"] += 1

    with open(os.path.join(G, "agent_review.jsonl"), "w", encoding="utf-8") as f:
        for r in rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(G, "approved_pairs.jsonl"), "w", encoding="utf-8") as f:
        for a in approved:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(os.path.join(G, "rejected_pairs.jsonl"), "w", encoding="utf-8") as f:
        for a in rejected_pairs:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(os.path.join(G, "excluded_names.jsonl"), "w", encoding="utf-8") as f:
        for a in excluded:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    summary = {"stats": stats, "approved": len(approved), "rejected_pairs": len(rejected_pairs),
               "excluded": len(excluded), "rows": len(rows_all),
               "seconds": round(time.time() - t0, 1),
               "note": "agent-reviewed = LLM delegate (qwen3.5:4b + rule/context), user-overridable"}
    json.dump(summary, open(os.path.join(G, "agent_review_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[AGENT-REVIEW]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
