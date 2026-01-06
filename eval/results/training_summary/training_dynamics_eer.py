import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib import rcParams
import numpy as np

# ---------- Style ----------
rcParams.update({
    "font.family": 'serif',
    "font.size": 11,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
})

def norm(v):
    return v if v <= 1 else v

# ---------- X labels (500 interval) ----------
# 生成 0 到 2600，步长 500 的标签: [0, 500, 1000, 1500, 2000, 2500]
labels = list(range(0, 2601, 500))
x = list(range(len(labels)))

# ---------- Data (Interpolate) ----------
L1_raw = [0.1770, 0.0511, 0.0447]
L2_raw = [0.1929, 0.0464, 0.0417]
L3_raw = [0.1401, 0.0354, 0.0319]
L4_raw = [0.1215, 0.0408, 0.0355]
original_x = [0, 1249, 2594]

L1 = list(map(norm, L1_raw))
L2 = list(map(norm, L2_raw))
L3 = list(map(norm, L3_raw))
L4 = list(map(norm, L4_raw))

# 根据新的 labels 对数据进行线性插值
L1_interp = np.interp(labels, original_x, L1)
L2_interp = np.interp(labels, original_x, L2)
L3_interp = np.interp(labels, original_x, L3)
L4_interp = np.interp(labels, original_x, L4)

series = [L1_interp, L2_interp, L3_interp, L4_interp]
names  = ["bge-large-en-v1.5", "e5-large-v2", "multilingual-e5-large", "qwen3-embedding-0.6b"]

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor("#eef7f1")

all_vals = [v for arr in series for v in arr]
base_y = min(all_vals) - 0.01
top_y  = max(all_vals) + 0.01

colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])

for (y, name, c) in zip(series, names, colors):
    # 阴影
    ax.fill_between(x, y, base_y, facecolor=c, alpha=0.15, zorder=1)
    # 线条 + 发光效果
    (ln,) = ax.plot(x, y, color=c, linewidth=3.8, label=name, zorder=3)
    ln.set_path_effects([
        pe.Stroke(linewidth=8, foreground=c, alpha=0.35),
        pe.Normal(),
    ])

# 轴网格与边框
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(2.0)
ax.spines["bottom"].set_linewidth(2.0)

# ---------- 设置 X 轴 ----------
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)

# 设置显示范围，负值越小，0点离Y轴越远；这里设为 -0.1 比较紧凑
ax.set_xlim(-0.1, len(x)-0.9)
ax.set_ylim(base_y, top_y)

ax.tick_params(axis='x', labelsize=27)
ax.tick_params(axis='y', labelsize=27)
ax.set_ylabel("EER", fontsize=32)
ax.set_xlabel("Training Steps", fontsize=32)

ax.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=20)

fig.tight_layout()
fig.savefig("eer.png", dpi=300, bbox_inches="tight")
fig.savefig("eer.pdf", bbox_inches="tight")
plt.show()