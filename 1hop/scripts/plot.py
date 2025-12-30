import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
layers = np.arange(12)

# AlignSAE 数据
align_eff_feat = [5.706, 3.239, 1.997, 2.627, 2.601, 1.571, 1.809, 1.339, 1.046, 1.053, 1.055, 1.001]
align_top1_conc = [0.253, 0.520, 0.753, 0.637, 0.714, 0.893, 0.844, 0.936, 0.991, 0.990, 0.990, 1.000]

# Traditional SAE 数据
trad_eff_feat = [743.873, 600.364, 275.956, 474.001, 501.939, 589.424, 585.605, 632.580, 732.881, 975.815, 1497.367, 2004.418]
trad_top1_conc = [0.006, 0.017, 0.027, 0.007, 0.007, 0.008, 0.012, 0.016, 0.018, 0.017, 0.010, 0.008]

# 2. 设置样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,          # 保持大字体
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.2
})

# 设置画布大小 (宽度适中，高度预留给顶部图例)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

color_align = '#2ca02c'  # 绿色
color_trad = '#d62728'   # 红色
lw = 3.5  # 线条粗细

# --------------------------
# 左图: EffFeat
# --------------------------
# 注意：移除了 marker 参数
l1, = ax1.plot(layers, align_eff_feat, color=color_align, linewidth=lw, label='AlignSAE')
l2, = ax1.plot(layers, trad_eff_feat, color=color_trad, linewidth=lw, label='Traditional SAE')

ax1.set_ylabel('EffFeat')
ax1.set_xlabel('Layer Index')
ax1.set_yscale('log')
ax1.set_xticks(layers[::2]) # 隔一个显示一个刻度，避免拥挤

# --------------------------
# 右图: Top1C
# --------------------------
ax2.plot(layers, align_top1_conc, color=color_align, linewidth=lw)
ax2.plot(layers, trad_top1_conc, color=color_trad, linewidth=lw)

ax2.set_ylabel('Top1C')
ax2.set_xlabel('Layer Index')
ax2.set_xticks(layers[::2])
ax2.set_ylim(0, 1.1)

# --------------------------
# 通用美化
# --------------------------
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.7)

# --------------------------
# 全局图例 (Global Legend)
# --------------------------
# 将图例放在两个图的上方正中间，横向排列 (ncol=2)
# 这样不会遮挡任何数据，也更符合论文排版
fig.legend(handles=[l1, l2], 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.1), # 稍微向上偏移，放在图片外面
           ncol=2, 
           frameon=False) 

plt.tight_layout()

# 3. 保存
# bbox_inches='tight' 会确保外部的图例也被包含进去，不会被切掉
plt.savefig('clean_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('clean_plot.png', dpi=300, bbox_inches='tight')

# plt.show()