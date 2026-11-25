"""queue_export: human-review queue CSV for canonicalization.
Combines: pending tier-3 pairs (sim in band, unprocessed), LLM-rejected pairs,
guard-blocked pairs, and CONFLICT-class canonical entities.
Output: ~/vicivil_tkg/human_queue.csv (semicolon-separated, Excel-friendly UTF-8-BOM).
"""
import csv
import json
import os

G = os.path.expanduser("~/vicivil_tkg/graph")
OUT = os.path.expanduser("~/vicivil_tkg/human_queue.csv")


def jl(p):
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    rows = []
    for p in jl(os.path.join(G, "pending_pairs.jsonl")):
        rows.append(["pending_tier3", round(p["sim"], 4), p["name_a"], p["name_b"],
                     p["class"], "merge?" ])
    for m in jl(os.path.join(G, "merge_log.jsonl")):
        ev = m.get("evidence", {})
        # tier1 stores 'variants' (readable names) — never fall back to md5 ids
        names = ev.get("names") or ev.get("variants") or [m.get("kept"), m.get("merged")]
        rows.append([m["method"], m.get("confidence", ""), str(names[0]) if names else "",
                     str(names[1]) if len(names) > 1 else "", "", "review"])
    for e in jl(os.path.join(G, "nodes_entity.jsonl")):
        if str(e.get("class", "")).startswith("CONFLICT") or e.get("member_count", 1) > 4:
            rows.append(["conflict_or_large_cluster", "", e["preferred_label"],
                         " | ".join(e["aliases"][:5]), e["class"], "check"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["queue_type", "score", "name_a", "name_b", "class", "action"])
        w.writerows(rows)
    by_type = {}
    for r in rows:
        by_type[r[0]] = by_type.get(r[0], 0) + 1
    print(f"human_queue.csv: {len(rows)} rows | {by_type}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
