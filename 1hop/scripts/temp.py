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

# 2. 设置绘图风格
# 设置全局字体为衬线体 (接近论文风格)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2
})

# 创建画布 (1行2列)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 定义颜色和标记 (参考原图)
color_align = '#2ca02c'  # 绿色
color_trad = '#d62728'   # 红色
marker_align = 'o'       # 圆点
marker_trad = 's'        # 方块
lw = 2                   # 线宽
ms = 7                   # 标记大小

# --------------------------
# 左图: Effective Features
# --------------------------
ax1.plot(layers, align_eff_feat, marker=marker_align, color=color_align, 
         linewidth=lw, markersize=ms, label='AlignSAE')
ax1.plot(layers, trad_eff_feat, marker=marker_trad, color=color_trad, 
         linewidth=lw, markersize=ms, label='Traditional SAE')

ax1.set_ylabel('Effective Features', fontsize=14)
ax1.set_xlabel('Layer Index', fontsize=14)
ax1.set_yscale('log') # 关键：原图左侧显然是Log Scale
ax1.set_xticks(layers)

# --------------------------
# 右图: Top-1 Concentration
# --------------------------
ax2.plot(layers, align_top1_conc, marker=marker_align, color=color_align, 
         linewidth=lw, markersize=ms, label='AlignSAE')
ax2.plot(layers, trad_top1_conc, marker=marker_trad, color=color_trad, 
         linewidth=lw, markersize=ms, label='Traditional SAE')

ax2.set_ylabel('Top-1 Concentration', fontsize=14)
ax2.set_xlabel('Layer Index', fontsize=14)
ax2.set_xticks(layers)

# 设置Y轴范围以美化显示 (根据新数据范围调整)
ax2.set_ylim(0, 1.1) 

# --------------------------
# 通用美化 (应用于两个子图)
# --------------------------
for ax in [ax1, ax2]:
    # 去除上方和右侧的边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加水平虚线网格
    ax.grid(axis='y', linestyle=':', linewidth=1, alpha=0.6)
    
    # 添加图例 (无边框)
    ax.legend(frameon=False, loc='best')

# 调整布局以防止重叠
plt.tight_layout()

# 显示图表
# plt.show()

# 如果需要保存图片，取消下面这行的注释
plt.savefig('sae_comparison.png', dpi=300, bbox_inches='tight')
# 如果需要保存图片，取消下面这行的注释
plt.savefig('sae_comparison.pdf', bbox_inches='tight')