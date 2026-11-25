"""extract_llm: P3 — VCLO-constrained LLM triple extraction (KaLLM recipe).
Sample chunks -> qwen3.5:4b extracts triples with predicate restricted to the VCLO
typed subset -> deterministic prune: predicate must be in the allowed set AND
subject/object (folded) must appear verbatim in the chunk text.
Reports raw vs kept (hallucination rate). Output: graph/llm_extraction.jsonl + stats.
"""
import json
import os
import random
import re
import time
import urllib.request

from .textops import fold_text, map_relation, is_literal
from .ontology import extraction_predicates

G = os.path.expanduser("~/vicivil_tkg/graph")
WS = os.path.expanduser("~/law_workspace")
MODEL = "qwen3.5:4b"
N_CHUNKS = 100

# DERIVED from vclo2_core.ttl (single source of truth) — no hardcoded predicate list
ALLOWED = [f"vclo:{name}" for name in extraction_predicates()]
ALLOWED_FOLD = [fold_text(a.replace("vclo:", "")) for a in ALLOWED]


def ollama(prompt, timeout=120):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "format": "json",
                       "think": False, "options": {"temperature": 0, "num_predict": 300}}).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def main():
    chunks = []
    with open(os.path.join(WS, "runs", "civil_chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["law_id"] == "BLDS2015" and c.get("chars", 0) > 150 and len(c.get("content", "")) < 1500:
                chunks.append(c)
    random.seed(20261125)
    random.shuffle(chunks)
    chunks = chunks[:N_CHUNKS]
    print(f"[EXTRACT] sample={len(chunks)} chunks (BLDS2015, 150<=chars<1500)")

    raw_n = kept_n = pruned_hallucination = pruned_other = 0
    out_rows = []
    t0 = time.time()
    for i, c in enumerate(chunks):
        prompt = (
            "Bạn là chuyên gia trích xuất triple pháp luật. Từ đoạn luật sau, trích TỐI ĐA 3 triple.\n"
            "Quan hệ CHỈ được chọn trong danh sách: "
            + ", ".join(a.replace("vclo:", "") for a in ALLOWED) + "\n"
            "Chủ thể/đối tượng PHẢI là cụm từ xuất hiện nguyên văn trong đoạn.\n"
            f'Đoạn ({c["chunk_id"]}): "{c["content"][:900]}"\n'
            'Trả lời JSON: {"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}'
        )
        try:
            resp = ollama(prompt)
            data = json.loads(resp)
            triples = data.get("triples", [])[:3]
        except Exception as e:
            triples = []
        ftext = fold_text(c["content"])
        for t in triples:
            s, p, o = str(t.get("subject", "")), str(t.get("predicate", "")), str(t.get("object", ""))
            raw_n += 1
            row = {"chunk_id": c["chunk_id"], "subject": s, "predicate": p, "object": o,
                   "status": "kept"}
            pf = fold_text(p)
            if pf not in ALLOWED_FOLD:
                row["status"] = "pruned_predicate_not_in_ontology"
                pruned_other += 1
            elif is_literal(s) or is_literal(o):
                row["status"] = "pruned_literal_endpoint"
                pruned_other += 1
            elif fold_text(s) not in ftext or fold_text(o) not in ftext:
                row["status"] = "pruned_hallucination_not_in_text"
                pruned_hallucination += 1
            else:
                kept_n += 1
            out_rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"[EXTRACT] {i+1}/{len(chunks)} raw={raw_n} kept={kept_n} "
                  f"({time.time()-t0:.0f}s)")
    stats = {"chunks": len(chunks), "raw_triples": raw_n, "kept": kept_n,
             "pruned_hallucination": pruned_hallucination,
             "pruned_other": pruned_other,
             "hallucination_rate": round(pruned_hallucination / max(1, raw_n), 4),
             "keep_rate": round(kept_n / max(1, raw_n), 4),
             "model": MODEL, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(G, "llm_extraction.jsonl"), "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(stats, open(os.path.join(G, "llm_extraction_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[EXTRACT]", stats)


if __name__ == "__main__":
    main()
