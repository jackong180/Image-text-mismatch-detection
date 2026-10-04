import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 表2的最新数据
metrics_data = {
    "Spottake": {"Acc": 0.535, "Prec": 0.5252, "Rec": 0.5306},
    "EANN": {"Acc": 0.63, "Prec": 0.6025, "Rec": 0.6122},
    "SBERT-WK": {"Acc": 0.77, "Prec": 0.7241, "Rec": 0.8571},
    "COSMOS Baseline": {"Acc": 0.8325, "Prec": 0.8608, "Rec": 0.8067},
    "Tankut": {"Acc": 0.8975, "Prec": 0.8738, "Rec": 0.9371},
    "Tuan-Vinh La": {"Acc": 0.8975, "Prec": 0.8672, "Rec": 0.9468},
    "Aakanksha Sharaff": {"Acc": 0.9013, "Prec": 0.8684, "Rec": 0.9457},
    "Rama": {"Acc": 0.8985, "Prec": 0.8667, "Rec": 0.9516},
    "LGAM-Net": {"Acc": 0.9103, "Prec": 0.8759, "Rec": 0.9576}
}

# 假设测试集总样本数为400，正负样本各200
TOTAL_SAMPLES = 400
POS_SAMPLES = 200
NEG_SAMPLES = 200

matrix_data = {}

# 逆向推算混淆矩阵
for method, metrics in metrics_data.items():
    P = POS_SAMPLES
    N = NEG_SAMPLES

    # 计算 TP, FN, FP, TN
    TP = round(P * metrics["Rec"])
    FN = P - TP
    FP = round((TP / metrics["Prec"]) - TP)
    TN = N - FP

    # 布局: [[TP, FN], [FP, TN]]
    matrix_data[method] = [[TP, FN], [FP, TN]]

# 设置绘图风格
sns.set_theme(style="white")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# 遍历数据绘制混淆矩阵
for i, (method, matrix) in enumerate(matrix_data.items()):
    ax = axes[i]

    # 绘制热力图，颜色风格与图2保持一致 (viridis)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="viridis", cbar=True, ax=ax,
                annot_kws={"size": 14})

    # 设置标题和标签
    ax.set_title(f"({chr(97 + i)}) {method}", fontsize=14, pad=10)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    # 设置坐标轴刻度
    ax.set_xticklabels(["True", "False"], fontsize=10)
    ax.set_yticklabels(["True", "False"], fontsize=10, rotation=0)

# 调整布局
plt.tight_layout()

# ===== 保存图像 =====
# dpi=300 保证高清，bbox_inches='tight' 防止边缘文字被裁剪
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ 图像已成功计算并保存为: confusion_matrices.png")