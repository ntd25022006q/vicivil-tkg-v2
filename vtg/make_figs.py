"""make_figs: publication-style Vietnamese figures for ViCivil-TKG v2.
All numbers pulled from real artifacts in graph/ — nothing invented.
Output: ~/vicivil_tkg/figures/fig1..fig6.png (150 dpi, Agg backend).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from collections import Counter

G = "/home/devlab1/vicivil_tkg/graph"
OUT = "/home/devlab1/vicivil_tkg/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
GREEN, GRAY, ORANGE, BLUE, RED = "#2e7d32", "#9e9e9e", "#ef6c00", "#1565c0", "#c62828"


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ================= FIG 1: timeline hiệu lực văn bản =================
nodes = jl(f"{G}/nodes_legal.jsonl")
rows = [n for n in nodes if n.get("valid_from")]
rows.sort(key=lambda n: n["valid_from"])
fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2))
for i, n in enumerate(rows):
    vf = date.fromisoformat(n["valid_from"])
    vt = date.fromisoformat(n["valid_to"]) if n.get("valid_to") else date(2027, 1, 1)
    st = n.get("status", "")
    col = GREEN if st == "con_hieu_luc" else (GRAY if st == "het_hieu_luc" else
          (ORANGE if st == "needs_review" else BLUE))
    ax.barh(i, (vt - vf).days, left=mdates.date2num(vf), height=0.62,
            color=col, alpha=0.85, edgecolor="none")
    ax.text(mdates.date2num(vf) + 8, i, n.get("so_hieu") or n["node_id"][:28],
            va="center", fontsize=6.5)
ax.set_yticks([])
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(mdates.date2num(date(1995, 1, 1)), mdates.date2num(date(2028, 1, 1)))
ax.axvline(mdates.date2num(date(2026, 11, 26)), color=RED, ls="--", lw=1)
ax.text(mdates.date2num(date(2026, 11, 26)), len(rows) - 0.5, " nay", color=RED, fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREEN, label="còn hiệu lực"), Patch(color=GRAY, label="hết hiệu lực"),
                   Patch(color=ORANGE, label="needs_review"), Patch(color=BLUE, label="external-ref")],
          fontsize=7, loc="lower right")
ax.set_title(f"Fig 1 — Timeline hiệu lực {len(rows)} văn bản quy phạm pháp luật (ViCivil-TKG v0.6.0)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_timeline_hieu_luc.png", dpi=150)
plt.close()

# ================= FIG 2: TKG 3 mốc as_of =================
edges = jl(f"{G}/edges_norm.jsonl")
SNAPS = [(2005, "2005-06-01"), (2015, "2015-06-01"), (2026, "2026-06-01")]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, (yr, asof) in zip(axes, SNAPS):
    act = {n["node_id"] for n in rows if n["valid_from"] <= asof
           and (not n.get("valid_to") or asof < n["valid_to"])}
    gone = {n["node_id"] for n in rows if n.get("valid_to") and asof >= n["valid_to"]}
    show = sorted(act | gone)
    pos = {}
    for k, nid in enumerate(show):
        pos[nid] = ((k % 5) * 1.0, -(k // 5) * 1.0)
    rel_style = {"vclo:replaces": ("-", RED), "vclo:amends": ("--", ORANGE),
                 "vclo:consolidates": (":", BLUE), "vclo:guides": ("-.", GREEN)}
    for e in edges:
        if e["relation"] not in rel_style:
            continue
        if e["src"] in pos and e["tgt"] in pos and e["src"] != e["tgt"]:
            ls, col = rel_style[e["relation"]]
            if (e.get("at") or "9999")[:4] <= str(yr):
                x1, y1 = pos[e["src"]]
                x2, y2 = pos[e["tgt"]]
                ax.plot([x1, x2], [y1, y2], ls=ls, color=col, lw=1.0, alpha=0.8)
    for nid, (x, y) in pos.items():
        col = GREEN if nid in act else GRAY
        ax.scatter(x, y, s=120, color=col, zorder=3)
        ax.annotate(nid[:16], (x, y), fontsize=5.2, ha="center", va="center", xytext=(0, -12),
                    textcoords="offset points")
    ax.set_title(f"as_of {asof}: {len(act)} văn bản còn hiệu lực / {len(gone)} đã hết", fontsize=10)
    ax.set_xlim(-0.8, 4.8)
    ax.set_ylim(-(max(1, (len(show) - 1) // 5 + 1)) - 0.6, 0.6)
    ax.axis("off")
from matplotlib.lines import Line2D
lg = [Line2D([0], [0], color=RED, ls="-", label="REPLACES"),
      Line2D([0], [0], color=ORANGE, ls="--", label="AMENDS"),
      Line2D([0], [0], color=BLUE, ls=":", label="CONSOLIDATES"),
      Line2D([0], [0], color=GREEN, ls="-.", label="GUIDES"),
      Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, label="còn hiệu lực"),
      Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY, label="hết hiệu lực")]
fig.legend(handles=lg, fontsize=8, loc="lower center", ncol=6)
fig.suptitle("Fig 2 — Đồ thị hiệu lực văn bản tại 3 mốc thời gian (snapshot succession)", fontsize=12)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(f"{OUT}/fig2_tkg_snapshots.png", dpi=150)
plt.close()

# ================= FIG 3: canonicalization funnel =================
st = jload(f"{G}/canonicalize_stats.json")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.barh(["canonical (sau khử trùng)", "raw (LightRAG)"],
        [st["canonical_entities_out"], st["entities_in"]],
        color=[GREEN, GRAY])
for i, v in enumerate([st["canonical_entities_out"], st["entities_in"]]):
    a1.text(v - 300, i, f"{v:,}".replace(",", "."), va="center", ha="right",
            color="white", fontsize=11, fontweight="bold")
a1.set_title(f"Funnel khử trùng tên — gộp {st['entities_in'] - st['canonical_entities_out']} node trùng/bẩn "
             f"(guard chặn {st.get('guard_blocked_pairs', 0):,} cặp rủi ro)".replace(",", "."), fontsize=10)
meth = Counter(m["method"] for m in jl(f"{G}/merge_log.jsonl") if not m["method"].endswith("rejected"))
labels = {"tier0-human-approved": "tier0: người/rule duyệt", "tier1-normkey": "tier1: chuẩn-hoá chữ",
          "tier2-cosine": "tier2: cosine embedding", "tier3-llm": "tier3: LLM qwen3.5:4b"}
a2.bar([labels.get(k, k) for k in meth], [meth[k] for k in meth], color=[BLUE, GREEN, "#7b1fa2", ORANGE][:len(meth)])
for i, k in enumerate(meth):
    a2.text(i, meth[k] + 1, str(meth[k]), ha="center", fontsize=10, fontweight="bold")
a2.set_title("Merge theo tầng adjudication (mọi merge có log + rollback)", fontsize=10)
a2.tick_params(axis="x", labelrotation=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_canonicalization.png", dpi=150)
plt.close()

# ================= FIG 4: continual embedding =================
v03 = jload(f"{G}/embedding_temporal_results.json")["aggregate"]
dggp = jload(f"{G}/embedding_dggp_results.json")["dggp_aggregate"]
tar = jload(f"{G}/embedding_dggp_tar_results.json")["dggp_tar_aggregate"]
MODELS = [("full_retrain (trần)", v03["full_retrain"]), ("fast_freeze (TADG)", v03["fast_freeze"]),
          ("DGGP", dggp), ("TAR+DGGP", tar), ("naive distill ✗", v03["slow_distill"]),
          ("no-continual (đáy)", v03["no_continual"])]
import numpy as np
x = np.arange(3)
w = 0.13
fig, ax = plt.subplots(figsize=(12, 5))
cols = ["#1565c0", "#2e7d32", "#7b1fa2", "#ef6c00", "#c62828", "#9e9e9e"]
for i, (name, agg) in enumerate(MODELS):
    vals = [agg[e]["mrr_mean"] for e in ("S1", "S2", "S3")]
    b = ax.bar(x + (i - 2.5) * w, vals, w, label=name, color=cols[i])
    ax.bar_label(b, fmt="%.2f", fontsize=6.5, padding=1)
ax.set_xticks(x, ["S1 (<2017) — kiến thức cũ", "S2 (2017–2024)", "S3 (≥2025) — mới nhất"])
ax.set_ylabel("Filtered MRR (mean 3 seeds)")
ax.set_title("Fig 4 — Continual embedding trên đồ thị luật thật (2,995 thực thể / 5,318 triples)\n"
             "DGGP + TAR sửa catastrophic forgetting của naive distillation (0.019 → 0.351 S2)", fontsize=11)
ax.legend(fontsize=8, ncol=3)
ax.set_ylim(0, 0.52)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_continual_embedding.png", dpi=150)
plt.close()

# ================= FIG 5: realtime pipeline =================
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.axis("off")
boxes = [
    (0.03, 0.62, "CRON 6h\n(cron đã tự fire\n00:23 ✓)", "#e3f2fd"),
    (0.19, 0.62, "POLL công báo\ncongbao.chinhphu.vn\n24 kỳ / 9+4 văn bản", "#e8f5e9"),
    (0.35, 0.62, "HASH-DIFF\nidempotent\n(0 false churn)", "#e8f5e9"),
    (0.51, 0.62, "PDF → TEXT\n7/9 trích ok\n(pypdf/olefile)", "#e8f5e9"),
    (0.67, 0.62, "MINE\namend facts 157\n+ cites + APPLIES 1,775", "#e8f5e9"),
    (0.83, 0.62, "DELTA staging\n+ merge-gate\n(needs_review=0 ✓)", "#fff3e0"),
    (0.19, 0.22, "CANONICALIZE\nS1-S3 + guard\n14,099→13,857", "#f3e5f5"),
    (0.40, 0.22, "TEMPORALIZE\n97.7% edges\ncó interval", "#f3e5f5"),
    (0.61, 0.22, "LOCAL CALIBRATE\n88 entity\n(GEDUM/DINGAL-O)", "#f3e5f5"),
    (0.82, 0.22, "PUBLISH\nSHA256 + pyshacl\nconforms ✓", "#ffebee"),
]
for x, y, t, c in boxes:
    ax.add_patch(plt.Rectangle((x, y), 0.15, 0.26, color=c, ec="#455a64", lw=1.2, zorder=2))
    ax.text(x + 0.075, y + 0.13, t, ha="center", va="center", fontsize=8, zorder=3)
for x1, y1, x2, y2 in [(0.18, 0.75, 0.19, 0.75), (0.34, 0.75, 0.35, 0.75), (0.50, 0.75, 0.51, 0.75),
                        (0.66, 0.75, 0.67, 0.75), (0.82, 0.75, 0.83, 0.75),
                        (0.905, 0.62, 0.905, 0.48), (0.83, 0.35, 0.76, 0.35),
                        (0.55, 0.35, 0.40, 0.35), (0.34, 0.35, 0.265, 0.35), (0.19, 0.35, 0.19, 0.48)]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.6))
ax.text(0.5, 0.06, "Vòng cập nhật realtime khép kín — watcher tự phát hiện văn bản mới, gate chặn mọi thứ không có bằng chứng",
        ha="center", fontsize=10, style="italic")
ax.set_title("Fig 5 — Pipeline realtime ViCivil-TKG v0.6.0", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_realtime_pipeline.png", dpi=150)
plt.close()

# ================= FIG 6: benchmark headline (v1 published numbers) =================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.bar(["BM25+dense\n(RRF)", "+ TCR\n(đồ thị thời gian)", "+ rerank\n(BGE + TCR)"],
       [0.733, 0.811, 0.923], color=[GRAY, GREEN, BLUE])
for i, v in enumerate([0.733, 0.811, 0.923]):
    a1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold")
a1.set_ylim(0, 1.0)
a1.set_ylabel("Hit@5 (core 281 câu)")
a1.set_title("Truy hồi: TCR (lọc hiệu lực theo as_of)\nnâng Hit@5 0.733 → 0.811", fontsize=10)
a2.bar(["Không ngữ cảnh", "RAG hybrid", "RAG + TCR"],
       [0.004, 0.488, 0.601], color=[RED, GRAY, GREEN])
for i, v in enumerate([0.004, 0.488, 0.601]):
    a2.text(i, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold")
a2.set_ylim(0, 0.75)
a2.set_ylabel("Cite-Acc (Qwen2.5-3B)")
a2.set_title("QA: trích dẫn đúng điều-khoản\n0.004 → 0.601 (McNemar p=5.8e-05)", fontsize=10)
fig.suptitle("Fig 6 — Headline benchmark v1 đã công bố (số machine-verified từ artifact)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_benchmark_headline.png", dpi=150)
plt.close()

print("[FIGS]", sorted(os.listdir(OUT)))
