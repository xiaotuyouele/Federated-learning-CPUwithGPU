#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullLocator

# =========================================================
# 1. 全局设置
# =========================================================
SAVE_DIR = "/content/save"
os.makedirs(SAVE_DIR, exist_ok=True)

candidate_fonts = [
    'Microsoft YaHei',
    'SimSun',
    'SimHei',
    'KaiTi',
    'FangSong',
    'Arial Unicode MS'
]

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
chosen_font = None
for f in candidate_fonts:
    if f in available_fonts:
        chosen_font = f
        break

if chosen_font is None:
    chosen_font = 'DejaVu Sans'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [chosen_font]
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

plt.rcParams['grid.color'] = '#D0D0D0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7
plt.rcParams['grid.alpha'] = 0.6

plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = '#BFBFBF'

plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# =========================================================
# 2. IEEE 风格配色
# =========================================================
colors = [
    '#1F4E79',
    '#C0504D',
    '#6B8E23',
    '#7F60A8',
    '#B8860B',
    '#4F81BD',
    '#8C564B'
]

markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

# =========================================================
# 3. 读取数据
# =========================================================
convergence_path = os.path.join(SAVE_DIR, "convergence.npy")
alpha_results_path = os.path.join(SAVE_DIR, "results.npy")
frac_results_path = os.path.join(SAVE_DIR, "frac_results.npy")

if not os.path.exists(convergence_path):
    raise FileNotFoundError(f"未找到文件: {convergence_path}")
if not os.path.exists(alpha_results_path):
    raise FileNotFoundError(f"未找到文件: {alpha_results_path}")
if not os.path.exists(frac_results_path):
    raise FileNotFoundError(f"未找到文件: {frac_results_path}")

convergence = np.load(convergence_path, allow_pickle=True).item()
alpha_results = np.load(alpha_results_path, allow_pickle=True).item()
frac_results = np.load(frac_results_path, allow_pickle=True).item()

# =========================================================
# 4. 保存函数：只保存 PDF
# =========================================================
def save_pdf(fig, base_name):
    pdf_path = os.path.join(SAVE_DIR, f"{base_name}.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)
    print(f"已保存: {pdf_path}")

# 对数坐标刻度格式化：不用科学计数法
def plain_log_formatter(x, pos):
    if x >= 1:
        if abs(x - round(x)) < 1e-10:
            return str(int(round(x)))
        return f'{x:g}'
    else:
        return f'{x:g}'

# =========================================================
# 5. 收敛曲线
# =========================================================
fig = plt.figure(figsize=(7.2, 5.4))

for i, (alpha, curve) in enumerate(sorted(convergence.items())):
    curve = np.array(curve)
    x = np.arange(1, len(curve) + 1)

    plt.plot(
        x,
        curve,
        label=f'α={alpha}',
        color=colors[i % len(colors)],
        linestyle=linestyles[i % len(linestyles)],
        marker=markers[i % len(markers)],
        markevery=max(1, len(curve) // 6),
        linewidth=2.0,
        markersize=5.5,
        markerfacecolor='white',
        markeredgecolor=colors[i % len(colors)],
        markeredgewidth=1.0
    )

plt.xlabel('通信轮次（Epoch）', fontsize=14)
plt.ylabel('测试准确率（%）', fontsize=14)
plt.title('不同 α 的收敛曲线', fontsize=16, pad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='major')

leg = plt.legend(fontsize=10.5, loc='best', ncol=1)
for lh in leg.legend_handles:
    lh.set_linewidth(2.0)

plt.tight_layout()
save_pdf(fig, '收敛曲线_alpha_ieee')
plt.close(fig)

# =========================================================
# 6. α vs accuracy（横轴对数坐标 + 普通数字显示）
# =========================================================
alphas = sorted(alpha_results.keys())
means = [alpha_results[a][0] for a in alphas]
stds = [alpha_results[a][1] for a in alphas]

fig = plt.figure(figsize=(6.6, 5.2))

plt.plot(
    alphas,
    means,
    color='#1F4E79',
    linestyle='-',
    linewidth=2.0,
    marker='o',
    markersize=6.5,
    markerfacecolor='white',
    markeredgecolor='#1F4E79',
    markeredgewidth=1.4
)

plt.errorbar(
    alphas,
    means,
    yerr=stds,
    fmt='none',
    ecolor='#666666',
    elinewidth=1.4,
    capsize=4,
    capthick=1.4,
    zorder=1
)

plt.xlabel('α', fontsize=14)
plt.ylabel('最终测试准确率（%）', fontsize=14)
plt.title('α 与最终测试准确率关系', fontsize=16, pad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='major')

# 横轴使用对数坐标
plt.xscale('log')

# 只显示你自己的 alpha 值，不显示科学计数法
ax = plt.gca()
ax.set_xticks(alphas)
ax.xaxis.set_major_formatter(FuncFormatter(plain_log_formatter))
ax.xaxis.set_minor_locator(NullLocator())

plt.tight_layout()
save_pdf(fig, 'alpha_vs_accuracy_ieee')
plt.close(fig)

# =========================================================
# 7. frac vs accuracy
# =========================================================
alphas_frac = sorted(set([k[0] for k in frac_results.keys()]))
fracs = sorted(set([k[1] for k in frac_results.keys()]))

fig = plt.figure(figsize=(7.4, 5.6))

for i, alpha in enumerate(alphas_frac):
    means_curve, stds_curve = [], []

    for frac in fracs:
        mean, std = frac_results[(alpha, frac)]
        means_curve.append(mean)
        stds_curve.append(std)

    plt.errorbar(
        fracs,
        means_curve,
        yerr=stds_curve,
        label=f'α={alpha}',
        color=colors[i % len(colors)],
        linestyle=linestyles[i % len(linestyles)],
        marker=markers[i % len(markers)],
        linewidth=1.9,
        markersize=5.2,
        markerfacecolor='white',
        markeredgecolor=colors[i % len(colors)],
        markeredgewidth=1.1,
        elinewidth=1.2,
        capsize=3.5,
        capthick=1.2
    )

plt.xlabel('客户端参与比例（Fraction）', fontsize=14)
plt.ylabel('最终测试准确率（%）', fontsize=14)
plt.title('不同 α 下 frac 对准确率的影响', fontsize=16, pad=10)
plt.xticks(fracs, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='major')

plt.legend(
    fontsize=10,
    loc='best',
    ncol=1,
    borderpad=0.5,
    handlelength=2.6,
    labelspacing=0.4
)

plt.tight_layout()
save_pdf(fig, 'frac_vs_accuracy_multi_alpha_ieee')
plt.close(fig)

print("=" * 60)
print("绘图完成")
print(f"当前使用字体: {chosen_font}")
print(f"保存目录: {SAVE_DIR}")
print("=" * 60)
