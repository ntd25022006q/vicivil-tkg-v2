"""triage: final queue refinement — two jobs.
1) context re-adjudication: for REJECTED pairs lacking topology support, re-ask the LLM
   WITH the original chunk contexts (GPT-NER self-verification style). Merge only on
   "same" + confidence>=0.9 with context evidence. Rescues true-sames like 'kim khí quí/quý'.
2) queue triage: rewrite human_queue.csv into 4 priority sections so human effort lands
   only on genuinely-human rows:
   P1 accepted-merge WITHOUT topology support (risk of wrong merge)
   P2 conflicts / large clusters
   P3 rejected pairs (conservative-by-default; skim)
   P4 accepted-merge WITH topology support (informational, auto-OK)
Output: graph/context_readjudicated.jsonl + human_queue.csv (regenerated, sectioned).
"""
import json
import os
import time
import urllib.request

from .canonicalize import merge_blocked, ollama_same_entity

G = os.path.expanduser("~/vicivil_tkg/graph")
WS = os.path.expanduser("~/law_workspace")


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main(cap=150, queue_only=False):
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
                cid = c.get("chunk_id") or c.get("__id__") or ""
                chunk_text[str(cid)] = (c.get("content") or "")[:220]

    audit = jl(os.path.join(G, "merge_topology_audit.jsonl"))
    flagged_rejected = [r for r in audit
                        if r.get("topology") == "needs_topology_review"
                        and r["method"] == "tier3-llm-rejected"]

    # ---- 1. context re-adjudication (skippable via --queue-only, reloads prior results)
    saved_path = os.path.join(G, "context_readjudicated.jsonl")
    if queue_only and os.path.exists(saved_path):
        out = jl(saved_path)
        stats = {"rechecked": len(out), "rescued": sum(1 for r in out if r.get("context_decision") == "merge"),
                 "confirmed_rejected": sum(1 for r in out if r.get("context_decision") == "confirmed_rejected"),
                 "errors": 0}
        print("[CTX-READJUDICATE] --queue-only: loaded", len(out), "prior decisions")
    else:
        stats = {"rechecked": 0, "rescued": 0, "confirmed_rejected": 0, "errors": 0}
        out = []
    t0 = time.time()
    for r in ([] if queue_only else flagged_rejected[:cap]):
        names = (r.get("evidence") or {}).get("names") or ["", ""]
        na, nb = names[0], names[1]
        if merge_blocked(na, nb):
            continue
        ea = by_alias.get(na) or {}
        eb = by_alias.get(nb) or {}
        ca = [chunk_text.get(cid, "") for cid in (ea.get("chunk_ids") or [])[:2]]
        cb = [chunk_text.get(cid, "") for cid in (eb.get("chunk_ids") or [])[:2]]
        ctx = " | ".join(x for x in ca + cb if x)[:500]
        prompt = (f"Hai thuật ngữ pháp luật: A=\"{na}\" và B=\"{nb}\".\n"
                  f"Ngữ cảnh xuất hiện: {ctx or '(không có ngữ cảnh lưu)'}\n"
                  "A và B có cùng một khái niệm/thực thể pháp luật không (chú ý biến thể chính tả "
                  "viết tắt như quí/quý)? Trả lời CHỈ JSON: {\"same\": true|false, \"confidence\": 0.0-1.0}")
        try:
            body = json.dumps({"model": "qwen3.5:4b", "prompt": prompt, "stream": False,
                               "format": "json", "think": False,
                               "options": {"temperature": 0, "num_predict": 40}}).encode("utf-8")
            req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                o = json.loads(resp.read().decode("utf-8")).get("response", "")
            same = json.loads(o) if o else {}
            ok = bool(same.get("same")) and float(same.get("confidence", 0)) >= 0.9
            stats["rechecked"] += 1
            if ok:
                stats["rescued"] += 1
                out.append({**r, "context_decision": "merge", "confidence": same.get("confidence")})
            else:
                stats["confirmed_rejected"] += 1
                out.append({**r, "context_decision": "confirmed_rejected"})
        except Exception as e:
            stats["errors"] += 1
    with open(os.path.join(G, "context_readjudicated.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[CTX-READJUDICATE]", stats, f"({time.time()-t0:.0f}s)")

    # ---- 2. queue triage (4 priority sections)
    supported = {tuple((r.get("evidence") or {}).get("names", ["", ""])[:2]) for r in audit
                 if r.get("topology") == "supported"}
    rescued = {tuple((r.get("evidence") or {}).get("names", ["", ""])[:2]) for r in out
               if r.get("context_decision") == "merge"}
    sections = {"P1_review_merge_no_support": [], "P2_conflicts": [],
                "P3_rejected_skim": [], "P4_auto_ok": []}
    for m in jl(os.path.join(G, "merge_log.jsonl")):
        ev = m.get("evidence", {})
        names = ev.get("names") or ev.get("variants") or [m.get("kept"), m.get("merged")]
        a, b = str(names[0] if names else ""), str(names[1]) if len(names) > 1 else ""
        key = (a, b)
        if m["method"].startswith("tier"):
            if key in rescued or any(a in k and b in k for k in rescued):
                sections["P4_auto_ok"].append([m["method"], m.get("confidence", ""), a, b, "", "auto-ok-context"])
            elif key in supported or any(a in s and b in s for s in supported):
                sections["P4_auto_ok"].append([m["method"], m.get("confidence", ""), a, b, "", "auto-ok-topology"])
            else:
                sections["P1_review_merge_no_support"].append(
                    [m["method"], m.get("confidence", ""), a, b, "", "HUMAN-REVIEW"])
    for e in ents:
        if str(e.get("class", "")).startswith("CONFLICT") or e.get("member_count", 1) > 4:
            sections["P2_conflicts"].append(["conflict", "", e["preferred_label"],
                                             " | ".join(e["aliases"][:5]), e["class"], "check"])
    for m in jl(os.path.join(G, "merge_log.jsonl")):
        if m["method"].endswith("rejected"):
            ev = m.get("evidence", {})
            names = ev.get("names") or ["", ""]
            sections["P3_rejected_skim"].append([m["method"], m.get("confidence", ""),
                                                 str(names[0])[:70], str(names[1])[:70], "", "skim"])
    import csv
    rows = []
    for k in ("P1_review_merge_no_support", "P2_conflicts", "P3_rejected_skim", "P4_auto_ok"):
        for r in sections[k]:
            rows.append([k] + r)
    with open(os.path.expanduser("~/vicivil_tkg/human_queue.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["priority", "method", "score", "name_a", "name_b", "class", "action"])
        w.writerows(rows)
    counts = {k: len(v) for k, v in sections.items()}
    json.dump({"context": stats, "queue_sections": counts},
              open(os.path.join(G, "triage_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[TRIAGE]", counts, "| total rows:", len(rows))


if __name__ == "__main__":
    import sys
    main(queue_only="--queue-only" in sys.argv)
