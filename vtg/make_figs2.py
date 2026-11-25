"""make_figs2: fig7-10 — entity temporal coverage, amendment facts, adjudication funnel, queue donut."""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = "/home/devlab1/vicivil_tkg/graph"
OUT = "/home/devlab1/vicivil_tkg/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
GREEN, GRAY, ORANGE, BLUE, RED, PURPLE = "#2e7d32", "#9e9e9e", "#ef6c00", "#1565c0", "#c62828", "#7b1fa2"


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def jload(p):
    return json.load(open(p, encoding="utf-8"))


# ---- FIG 7: entity temporal coverage donut
ts = jload(f"{G}/temporalize_stats.json")
fig, ax = plt.subplots(figsize=(6, 5.4))
ax.pie([ts["with_interval"], ts["without_interval"]],
       labels=[f"có interval\n{ts['with_interval']:,}".replace(",", "."),
               f"chưa có\n{ts['without_interval']}"],
       colors=[GREEN, GRAY], autopct="%1.1f%%", startangle=90,
       wedgeprops=dict(width=0.42), textprops={"fontsize": 10})
ax.set_title("Fig 7 — Cạnh thực thể mang khoảng hiệu lực\n(thừa kế cửa sổ văn bản nguồn)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_entity_temporal_coverage.png", dpi=150)
plt.close()

# ---- FIG 8: amendment facts theo luật đích + năm
facts = jl(f"{G}/article_amend_facts.jsonl")
by_tgt = Counter(f.get("target_law_id") or "chưa-xác-định" for f in facts)
by_year = Counter(f["amender_law_id"][-9:-5] if f.get("amender_law_id") and "/" in f.get("amender_law_id", "")
                  else "?" for f in facts)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
ks = list(by_tgt)
a1.barh(ks, [by_tgt[k] for k in ks], color=[GREEN if k != "chưa-xác-định" else GRAY for k in ks])
for i, k in enumerate(ks):
    a1.text(by_tgt[k] + 0.4, i, str(by_tgt[k]), va="center", fontsize=10, fontweight="bold")
a1.set_title(f"Amendment facts cấp ĐIỀU theo luật đích (n={len(facts)})", fontsize=10)
a1.set_xlabel("số điều bị trích là bị sửa đổi")
ys = sorted(y for y in by_year if y != "?")
a2.bar(ys, [by_year[y] for y in ys], color=BLUE)
a2.set_title("Phân bổ theo năm luật sửa đổi", fontsize=10)
a2.tick_params(axis="x", labelrotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_amendment_facts.png", dpi=150)
plt.close()

# ---- FIG 9: adjudication funnel
st = jload(f"{G}/canonicalize_stats.json")
stages = ["Band 0.80-0.92\ncần thẩm định", "Đã thẩm LLM\n(qwen3.5:4b)", "Guard chặn\n(nguy hiểm)", "Merge chấp nhận", "Rejected\n(conservative)"]
vals = [1151, 775, 33, 8 + 5 + 6 + 3, 295 + 472 + 81]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(stages, vals, color=[GRAY, BLUE, RED, GREEN, "#bdbdbd"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:,}".replace(",", "."), ha="center", fontweight="bold")
ax.set_ylabel("số cặp tên")
ax.set_title("Fig 9 — Phễu thẩm định tier-3 (mọi quyết định có log + rollback):\n"
             "775 cặp qua LLM + 114 qua ngữ cảnh (33 rescued / 81 confirmed) — 0 error", fontsize=10.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_adjudication_funnel.png", dpi=150)
plt.close()

# ---- FIG 10: queue triage donut
ts2 = jload(f"{G}/triage_stats.json")["queue_sections"]
labels = [f"P1 duyệt thật\n{ts2['P1_review_merge_no_support']}",
          f"P2 xung đột\n{ts2['P2_conflicts']}",
          f"P3 lướt\n{ts2['P3_rejected_skim']}",
          f"P4 tự động OK\n{ts2['P4_auto_ok']}"]
vals = [ts2["P1_review_merge_no_support"], ts2["P2_conflicts"],
        ts2["P3_rejected_skim"], ts2["P4_auto_ok"]]
fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.pie(vals, labels=labels, colors=[RED, ORANGE, GRAY, GREEN],
       autopct="%1.0f%%", startangle=120, wedgeprops=dict(width=0.42), textprops={"fontsize": 9.5})
ax.set_title("Fig 10 — Hàng chờ người (851 dòng, 4 mức ưu tiên)\nP1+P2 = 205 dòng là phán đoán pháp lý thật", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_queue_triage.png", dpi=150)
plt.close()

print("[FIGS2]", sorted(os.listdir(OUT)))
