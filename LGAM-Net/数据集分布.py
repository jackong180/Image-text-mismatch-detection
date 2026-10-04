import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 类别与对应百分比（按参考图录入）
categories = [
    "Automobile", "Brexit", "Celebrity", "Climate\nChange",
    "Covid", "Crime", "Education", "Elections",
    "Environment", "Global\nWarming", "Healthcare",
    "Immigration", "Politics", "Racism", "Space",
    "Sports", "Travel", "Vaccines", "Wildlife"
]

percentages = [
    2.4, 2.7, 10.7, 5.7, 7.0, 5.3, 6.6, 14.8,
    7.5, 4.6, 1.1, 1.7, 4.2, 7.637, 4.2, 6.3,
    3.0, 0.6, 4.1
]

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#000000",
    "axes.linewidth": 0.8,
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "savefig.dpi": 300,
})

fig, ax = plt.subplots(figsize=(12, 4.6))

x = np.arange(len(categories))

bars = ax.bar(
    x,
    percentages,
    width=0.70,
    color="#FFC0CB",
    edgecolor="#000000",
    linewidth=2.0,
    zorder=4
)

# 柱顶百分比：自动去掉多余的末尾零
for bar, value in zip(bars, percentages):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.12,
        f"{value:g}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#000000"
    )

ax.set_title(
    "Category Frequency Distribution of the Dataset",
    pad=12
)
ax.set_ylabel("% Images")
ax.set_ylim(0, 16)
ax.set_xlim(-0.6, len(categories) - 0.4)

ax.set_xticks(x)
ax.set_xticklabels(
    categories,
    rotation=48,
    ha="right",
    rotation_mode="anchor"
)
ax.yaxis.set_major_locator(MultipleLocator(4))

# 浅灰色水平网格
ax.set_axisbelow(True)
ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
ax.tick_params(axis="both", length=0)

fig.tight_layout()

# PNG用于插图，PDF/SVG便于排版与编辑
fig.savefig("category_distribution.png", bbox_inches="tight")

plt.show()