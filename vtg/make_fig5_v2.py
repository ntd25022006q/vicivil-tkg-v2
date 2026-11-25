"""make_fig5_v2: polished realtime pipeline diagram — serpentine layout, arrows ONLY in
gaps between boxes (no text collisions), rounded boxes, numbered stages."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "/home/devlab1/vicivil_tkg/figures"
os.makedirs(OUT, exist_ok=True)

BLUE, GREEN, ORANGE, PURPLE, RED, INK = "#1976d2", "#2e7d32", "#ef6c00", "#7b1fa2", "#c62828", "#263238"

ROW1 = [
    ("1", "CRON 6h", "tự fire 00:23 ✓", "#e3f2fd", BLUE),
    ("2", "POLL công báo", "congbao.chinhphu.vn\n24 kỳ · 13 văn bản", "#e8f5e9", GREEN),
    ("3", "HASH-DIFF", "idempotent\n0 false churn", "#e8f5e9", GREEN),
    ("4", "PDF → TEXT", "pypdf / olefile\n9/9 trích ok", "#e8f5e9", GREEN),
    ("5", "MINE", "amend 157 · cites\nAPPLIES 1,775", "#e8f5e9", GREEN),
    ("6", "DELTA + GATE", "needs_review = 0 ✓\n4/4 có ngày hiệu lực", "#fff3e0", ORANGE),
]
ROW2 = [  # flow: phải → trái
    ("7", "CANONICALIZE", "S1-S3 + guard\n14,099 → 13,873", "#f3e5f5", PURPLE),
    ("8", "TEMPORALIZE", "97.7% cạnh\ncó interval", "#f3e5f5", PURPLE),
    ("9", "LOCAL CALIBRATE", "88 entity\nGEDUM/DINGAL-O", "#f3e5f5", PURPLE),
    ("10", "PUBLISH", "SHA256 · pyshacl\nconforms ✓", "#ffebee", RED),
]

W, H, GAP, X0 = 0.145, 0.26, 0.025, 0.010
Y1, Y2 = 0.58, 0.16          # đáy hàng 1 / hàng 2
C1, C2 = Y1 + H / 2, Y2 + H / 2

fig, ax = plt.subplots(figsize=(14.5, 6.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

pos1, pos2 = [], []
for i, (num, title, sub, face, ec) in enumerate(ROW1):
    x = X0 + i * (W + GAP)
    pos1.append((x, x + W))
    ax.add_patch(FancyBboxPatch((x, Y1), W, H, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=face, ec=ec, lw=1.6, zorder=2))
    ax.text(x + W / 2, Y1 + H * 0.78, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK, zorder=3)
    ax.text(x + W / 2, Y1 + H * 0.38, sub, ha="center", va="center", fontsize=8.2, color=INK, zorder=3)
    ax.text(x + 0.011, Y1 + H - 0.030, num, fontsize=8.5, fontweight="bold", color=ec, zorder=3)
for i, (num, title, sub, face, ec) in enumerate(ROW2):
    x = 0.1755 + (3 - i) * (W + GAP)
    pos2.append((x, x + W))
    ax.add_patch(FancyBboxPatch((x, Y2), W, H, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=face, ec=ec, lw=1.6, zorder=2))
    ax.text(x + W / 2, Y2 + H * 0.78, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK, zorder=3)
    ax.text(x + W / 2, Y2 + H * 0.38, sub, ha="center", va="center", fontsize=8.2, color=INK, zorder=3)
    ax.text(x + 0.011, Y2 + H - 0.030, num, fontsize=8.5, fontweight="bold", color=ec, zorder=3)

ARROW = dict(arrowstyle="-|>", color=INK, lw=2.0, mutation_scale=16)
# hàng 1: mũi tên CHỈ trong khe giữa các hộp
for i in range(5):
    ax.annotate("", xy=(pos1[i + 1][0] - 0.002, C1), xytext=(pos1[i][1] + 0.002, C1),
                arrowprops=ARROW, zorder=4)
# rẽ xuống: DELTA (cuối hàng 1) → elbow → CANONICALIZE (đầu hàng 2)
xd1 = (pos1[5][0] + pos1[5][1]) / 2
xc2 = (pos2[0][0] + pos2[0][1]) / 2
ax.plot([xd1, xd1], [Y1, 0.475], color=INK, lw=2.0, zorder=1)
ax.plot([xd1, xc2], [0.475, 0.475], color=INK, lw=2.0, zorder=1)
ax.annotate("", xy=(xc2, Y2 + H + 0.002), xytext=(xc2, 0.475), arrowprops=ARROW, zorder=4)
# hàng 2: chảy phải → trái
for i in range(3):
    ax.annotate("", xy=(pos2[i + 1][1] + 0.002, C2), xytext=(pos2[i][0] - 0.002, C2),
                arrowprops=ARROW, zorder=4)

ax.text(0.5, 0.955, "Fig 5 — Pipeline cập nhật realtime ViCivil-TKG v0.7.0",
        ha="center", fontsize=13, fontweight="bold", color=INK)
ax.text(0.5, 0.90, "10 giai đoạn khép kín · chạy tự động mỗi 6 giờ · mọi kết quả có provenance + SHA256",
        ha="center", fontsize=9.5, color="#546e7a")
ax.text(0.5, 0.045, "Watcher tự phát hiện văn bản mới trên cổng công báo — gate chặn mọi thứ không có bằng chứng — không bao giờ đoán",
        ha="center", fontsize=9.5, style="italic", color="#455a64")
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_realtime_pipeline.png", dpi=150)
plt.close()
print("fig5 v2 written")
