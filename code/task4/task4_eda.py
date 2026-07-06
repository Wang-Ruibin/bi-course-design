#!/usr/bin/env python3
"""
任务4: 故障模式分析与可视化
分析生产线故障模式，生成8张分析图表
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ========== 中文字体设置 ==========
def setup_chinese_font():
    """设置matplotlib中文字体"""
    font_candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Source Han Sans SC",
        "AR PL UMing CN", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in font_candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用: {name}")
            return name
    # 尝试查找系统中文字体文件
    for f in fm.fontManager.ttflist:
        if any(kw in f.name.lower() for kw in ["hei", "song", "cjk", "chinese", "noto"]):
            plt.rcParams["font.sans-serif"] = [f.name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用: {f.name}")
            return f.name
    print("[字体] 警告: 未找到中文字体，中文可能无法正常显示")
    return None


# ========== 故障列定义 ==========
FAULT_COLS = [
    "物料推送装置故障1001",
    "物料检测装置故障2001",
    "填装装置检测故障4001",
    "填装装置定位故障4002",
    "填装装置填装故障4003",
    "加盖装置定位故障5001",
    "加盖装置加盖故障5002",
    "拧盖装置定位故障6001",
    "拧盖装置拧盖故障6002",
]

# 故障简称（用于图表标签）
FAULT_SHORT = {
    "物料推送装置故障1001": "推送故障\n(1001)",
    "物料检测装置故障2001": "检测故障\n(2001)",
    "填装装置检测故障4001": "填装检测\n(4001)",
    "填装装置定位故障4002": "填装定位\n(4002)",
    "填装装置填装故障4003": "填装故障\n(4003)",
    "加盖装置定位故障5001": "加盖定位\n(5001)",
    "加盖装置加盖故障5002": "加盖故障\n(5002)",
    "拧盖装置定位故障6001": "拧盖定位\n(6001)",
    "拧盖装置拧盖故障6002": "拧盖故障\n(6002)",
}

# 配色方案
PALETTE = sns.color_palette("Set2", 9)


def load_data():
    """加载并预处理数据"""
    path = os.path.join(os.path.dirname(__file__), "output", "task4_cleaned_data.csv")
    print(f"[数据] 加载: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[数据] 形状: {df.shape}")

    # 从秒数推导真实小时 (0-7，每3600秒一小时)
    df["hour"] = df["时间"] // 3600

    # 为故障列创建二值标记（原始值为故障代码，非0即有故障）
    for col in FAULT_COLS:
        df[col + "_bin"] = (df[col] > 0).astype(int)

    bin_cols = [c + "_bin" for c in FAULT_COLS]
    df["fault_count"] = df[bin_cols].sum(axis=1)

    return df


# ============================================================
# 图1: 故障类型分布饼图
# ============================================================
def plot_fault_distribution(df, outdir):
    """故障类型分布饼图"""
    print("[图1] 故障类型分布...")
    counts = {}
    for col in FAULT_COLS:
        counts[FAULT_SHORT[col]] = (df[col] > 0).sum()

    # 过滤掉频次为0的故障类型
    counts = {k: v for k, v in counts.items() if v > 0}
    labels = list(counts.keys())
    sizes = list(counts.values())
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%\n({int(p*total/100)})",
        startangle=90,
        colors=PALETTE[: len(labels)],
        textprops={"fontsize": 9},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("故障类型分布", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_fault_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存 01_fault_distribution.png (共{total}次故障)")


# ============================================================
# 图2: 各时段故障频次热力图
# ============================================================
def plot_fault_by_hour(df, outdir):
    """故障类型×小时热力图"""
    print("[图2] 各时段故障频次热力图...")
    # 构建故障类型×小时矩阵
    matrix = pd.DataFrame(index=[FAULT_SHORT[c] for c in FAULT_COLS])
    for h in sorted(df["hour"].unique()):
        hour_df = df[df["hour"] == h]
        col_vals = []
        for fc in FAULT_COLS:
            col_vals.append((hour_df[fc] > 0).sum())
        matrix[h] = col_vals

    matrix.columns = [f"{int(h)}:00" for h in matrix.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "故障次数"},
    )
    ax.set_title("各时段故障频次热力图", fontsize=16, fontweight="bold")
    ax.set_xlabel("时段", fontsize=12)
    ax.set_ylabel("故障类型", fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_fault_by_hour.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 保存 02_fault_by_hour.png")


# ============================================================
# 图3: 各生产线故障对比
# ============================================================
def plot_fault_by_line(df, outdir):
    """各生产线故障次数对比柱状图"""
    print("[图3] 各生产线故障对比...")
    lines = sorted(df["生产线编号"].unique())
    line_faults = {}
    for line in lines:
        line_df = df[df["生产线编号"] == line]
        line_faults[line] = (line_df["故障标记"] > 0).sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        list(line_faults.keys()),
        list(line_faults.values()),
        color=PALETTE[: len(line_faults)],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars, line_faults.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("各生产线故障次数对比", fontsize=16, fontweight="bold")
    ax.set_xlabel("生产线编号", fontsize=12)
    ax.set_ylabel("故障次数", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_fault_by_line.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存 03_fault_by_line.png ({len(lines)}条生产线)")


# ============================================================
# 图4: 故障持续时间箱线图
# ============================================================
def plot_fault_duration(df, outdir):
    """故障持续时间分析箱线图"""
    print("[图4] 故障持续时间分析...")

    # 找到故障标记为1的连续段，计算持续时间
    durations = {}
    for col in FAULT_COLS:
        bin_col = col + "_bin"
        # 找到连续故障段的起止
        series = df[bin_col].values
        dur_list = []
        i = 0
        while i < len(series):
            if series[i] == 1:
                j = i
                while j < len(series) and series[j] == 1:
                    j += 1
                dur_list.append(j - i)
                i = j
            else:
                i += 1
        if dur_list:
            durations[FAULT_SHORT[col]] = dur_list

    # 过滤掉没有故障记录的类型
    if not durations:
        print("  → 跳过: 无故障记录")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = list(durations.keys())
    data = [durations[k] for k in labels]

    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], PALETTE[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("各类故障持续时间分布（连续故障记录数）", fontsize=16, fontweight="bold")
    ax.set_xlabel("故障类型", fontsize=12)
    ax.set_ylabel("持续时间（记录数）", fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_fault_duration.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存 04_fault_duration.png ({len(labels)}种故障类型)")


# ============================================================
# 图5: 故障相关性热力图
# ============================================================
def plot_fault_correlation(df, outdir):
    """故障类型共现相关性热力图"""
    print("[图5] 故障相关性分析...")
    bin_cols = [c + "_bin" for c in FAULT_COLS]
    corr_df = df[bin_cols].copy()
    corr_df.columns = [FAULT_SHORT[c] for c in FAULT_COLS]
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "相关系数"},
    )
    ax.set_title("故障类型相关性矩阵", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_fault_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 保存 05_fault_correlation.png")


# ============================================================
# 图6: 按日故障趋势折线图
# ============================================================
def plot_fault_by_day(df, outdir):
    """按日故障趋势折线图（数据为单月，用日期代替月份）"""
    print("[图6] 每日故障趋势...")
    daily = df.groupby("日期")["故障标记"].sum().reset_index()
    daily.columns = ["日期", "故障次数"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["日期"], daily["故障次数"], marker="o", linewidth=2,
            color="#E74C3C", markersize=5)
    ax.fill_between(daily["日期"], daily["故障次数"], alpha=0.15, color="#E74C3C")

    # 标注均值线
    mean_val = daily["故障次数"].mean()
    ax.axhline(mean_val, color="#3498DB", linestyle="--", linewidth=1, alpha=0.7,
               label=f"日均故障: {mean_val:.0f}")
    ax.legend(fontsize=11)

    ax.set_title("每日故障次数趋势", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期（天）", fontsize=12)
    ax.set_ylabel("故障次数", fontsize=12)
    ax.set_xticks(daily["日期"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_fault_by_day.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → 保存 06_fault_by_day.png")


# ============================================================
# 图7: 白班vs夜班故障对比
# ============================================================
def plot_day_vs_night(df, outdir):
    """白班与夜班故障对比柱状图"""
    print("[图7] 白班vs夜班故障对比...")

    # is_daytime: 1=白班, 0=夜班
    day_df = df[df["is_daytime"] == 1]
    night_df = df[df["is_daytime"] == 0]

    day_counts = {}
    night_counts = {}
    for col in FAULT_COLS:
        short = FAULT_SHORT[col]
        day_counts[short] = (day_df[col] > 0).sum()
        night_counts[short] = (night_df[col] > 0).sum()

    labels = list(day_counts.keys())
    day_vals = [day_counts[k] for k in labels]
    night_vals = [night_counts[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, day_vals, width, label="白班 (is_daytime=1)",
                   color="#F39C12", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, night_vals, width, label="夜班 (is_daytime=0)",
                   color="#2C3E50", edgecolor="black", linewidth=0.5)

    ax.set_title("白班与夜班各故障类型对比", fontsize=16, fontweight="bold")
    ax.set_xlabel("故障类型", fontsize=12)
    ax.set_ylabel("故障次数", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "07_day_vs_night.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存 07_day_vs_night.png (白班{len(day_df)}条, 夜班{len(night_df)}条)")


# ============================================================
# 图8: 常见故障组合
# ============================================================
def plot_fault_combinations(df, outdir):
    """最常见故障组合柱状图"""
    print("[图8] 常见故障组合分析...")
    bin_cols = [c + "_bin" for c in FAULT_COLS]

    # 筛选至少有一个故障的行
    fault_rows = df[df["故障标记"] == 1].copy()

    # 构建故障组合字符串
    def get_combo(row):
        active = []
        for col in FAULT_COLS:
            if row[col + "_bin"] == 1:
                # 提取故障代码数字
                code = col[-4:]  # 如 "1001"
                active.append(code)
        return "+".join(sorted(active)) if active else "无"

    fault_rows = fault_rows.copy()
    fault_rows["combo"] = fault_rows.apply(get_combo, axis=1)

    combo_counts = fault_rows["combo"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(combo_counts))
    bars = ax.barh(range(len(combo_counts)), combo_counts.values, color=colors,
                   edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(combo_counts)))
    ax.set_yticklabels(combo_counts.index, fontsize=10)
    ax.invert_yaxis()

    for bar, val in zip(bars, combo_counts.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", ha="left", va="center", fontsize=10)

    ax.set_title("最常见故障类型组合 (Top 15)", fontsize=16, fontweight="bold")
    ax.set_xlabel("出现次数", fontsize=12)
    ax.set_ylabel("故障组合 (代码)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "08_fault_combinations.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存 08_fault_combinations.png ({len(combo_counts)}种组合)")


# ============================================================
# 主函数
# ============================================================
def main():
    # 设置字体
    setup_chinese_font()

    # 设置seaborn风格
    sns.set_style("whitegrid")

    # 输出目录
    outdir = os.path.join(os.path.dirname(__file__), "output", "task4_eda")
    os.makedirs(outdir, exist_ok=True)

    # 加载数据
    df = load_data()

    # 生成8张图
    plot_fault_distribution(df, outdir)
    plot_fault_by_hour(df, outdir)
    plot_fault_by_line(df, outdir)
    plot_fault_duration(df, outdir)
    plot_fault_correlation(df, outdir)
    plot_fault_by_day(df, outdir)
    plot_day_vs_night(df, outdir)
    plot_fault_combinations(df, outdir)

    # 验证输出
    files = sorted(os.listdir(outdir))
    pngs = [f for f in files if f.endswith(".png")]
    print(f"\n{'='*50}")
    print(f"[完成] 共生成 {len(pngs)} 张图表:")
    for f in pngs:
        fpath = os.path.join(outdir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
    print(f"{'='*50}")

    if len(pngs) < 8:
        print(f"警告: 预期至少8张图，实际生成{len(pngs)}张")
        sys.exit(1)


if __name__ == "__main__":
    main()
