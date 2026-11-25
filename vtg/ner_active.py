"""ner_active: GPT-NER (NAACL 2025) applied ACTIVELY on statute text.
Pass 1 (generation): qwen3.5:4b marks entities in @@ten [Loai]## format.
Pass 2 (self-verification): each marked entity is re-checked by the model
("is this a correct <Loai> entity in this context?") — kept only if confirmed.
GPT-NER's two signature mechanisms, executed for real. Output: graph/ner_active.jsonl + stats.
"""
import json
import os
import random
import re
import time
import urllib.request

from .textops import fold_text

G = os.path.expanduser("~/vicivil_tkg/graph")
WS = os.path.expanduser("~/law_workspace")
MODEL = "qwen3.5:4b"
N = 25
TYPES = "Law, Article, Court, StateOrgan, Person, Contract, Asset, TimePeriod"


def ollama(prompt, timeout=120, np_=400):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "think": False,
                       "options": {"temperature": 0, "num_predict": np_}}).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def main():
    chunks = []
    with open(os.path.join(WS, "runs", "civil_chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["law_id"] == "BLDS2015" and 200 < c.get("chars", 0) < 1200:
                chunks.append(c)
    random.seed(20261126)
    random.shuffle(chunks)
    chunks = chunks[:N]
    marked_total = verified_kept = verified_removed = empty = 0
    rows = []
    t0 = time.time()
    for c in chunks:
        p1 = (
            "Nhiệm vụ: đánh dấu thực thể pháp luật trong văn bản. Copy nguyên văn đoạn văn bản "
            "và chèn ký hiệu @@trước tên thực thể## ngay quanh tên, kèm [Loại] ngay sau tên, "
            f"Loại thuộc: {TYPES}.\n"
            "VÍ DỤ:\n"
            "Input: \"Theo Điều 455 Bộ luật Dân sự 2015, hợp đồng phải lập thành văn bản.\"\n"
            "Output: \"Theo @@455 [Article]## @@Bộ luật Dân sự 2015 [Law]##, @@hợp đồng [Contract]## phải lập thành văn bản.\"\n"
            "Chỉ xuất Output, không giải thích.\n"
            f'Đoạn: "{c["content"][:700]}"'
        )
        try:
            marked = ollama(p1)
        except Exception:
            marked = ""
        spans = re.findall(r"@@(.{1,60}?)\s*\[([A-Za-z ]{3,20})\]##", marked)
        marked_total += len(spans)
        verified = []
        for name, typ in spans:
            p2 = (f'Trong ngữ cảnh pháp luật sau: "{c["content"][:400]}"\n'
                  f'Cụm "{name.strip()}" có đúng là một thực thể loại {typ.strip()} không? '
                  'Trả lời CHỈ "yes" hoặc "no".')
            try:
                verdict = ollama(p2, np_=8).strip().lower()
            except Exception:
                verdict = ""
            if verdict.startswith("yes"):
                verified.append({"name": name.strip(), "type": typ.strip()})
                verified_kept += 1
            else:
                verified_removed += 1
        rows.append({"chunk_id": c["chunk_id"], "marked": len(spans),
                     "verified": verified,
                     "raw_head": marked[:150],
                     "text": c["content"][:200]})
        if not spans:
            empty += 1
    stats = {"chunks": N, "entities_marked": marked_total, "self_verified_kept": verified_kept,
             "self_verified_removed": verified_removed,
             "keep_rate": round(verified_kept / max(1, marked_total), 4),
             "chunks_with_no_marks": empty, "model": MODEL, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(G, "ner_active.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(stats, open(os.path.join(G, "ner_active_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[NER]", stats)


if __name__ == "__main__":
    main()
