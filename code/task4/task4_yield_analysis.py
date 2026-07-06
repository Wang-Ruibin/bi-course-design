#!/usr/bin/env python3
"""
task4_yield_analysis.py — 产量与合格率关系分析

功能：
1. 加载M301生产线数据（每日累计计数器取最大值）
2. 加载操作人员信息表
3. 计算产量（合格数+不合格数）和合格率（合格数/产量）
4. 按日分组统计分析
5. 单因素方差分析（ANOVA）：日期对产量/合格率的影响
6. 相关性分析
7. 可视化：箱线图、交互效应图、热力图

运行：python3 code/task4_yield_analysis.py（从项目根目录）
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头模式，服务器兼容
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_ROOT
    / "课程设计资料"
    / "泰迪杯2024年A题_说明及数据"
    / "A题-示例数据"
    / "附件3"
)
M301_FILE = DATA_DIR / "M301.csv"
OPERATOR_FILE = DATA_DIR / "操作人员信息表.xlsx"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
VIZ_DIR = OUTPUT_DIR / "task4_yield_viz"
ANALYSIS_CSV = OUTPUT_DIR / "task4_yield_analysis.csv"
ANOVA_CSV = OUTPUT_DIR / "task4_anova_results.csv"

# ============================================================
# 中文字体配置
# ============================================================
def setup_chinese_font() -> None:
    """设置matplotlib中文字体，按优先级尝试。"""
    font_candidates = [
        "SimHei",           # 黑体（Windows）
        "Microsoft YaHei",  # 微软雅黑（Windows）
        "WenQuanYi Micro Hei",  # 文泉驿（Linux）
        "Noto Sans CJK SC",     # Noto（Linux）
        "PingFang SC",           # 苹方（macOS）
        "STHeiti",               # 华文黑体（macOS）
    ]
    for font in font_candidates:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  使用字体: {font}")
            return
        except Exception:
            continue
    # 兜底：使用系统默认
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("  ⚠ 未找到中文字体，使用默认字体（中文可能显示为方块）")


# ============================================================
# 数据加载
# ============================================================
def load_production_data() -> pd.DataFrame:
    """
    加载M301.csv，提取每日最终累计值。
    合格数/不合格数为逐秒累计计数器，每日最大值即为当日总产量。
    """
    print("[1/6] 加载M301生产线数据...")
    df = pd.read_csv(M301_FILE, encoding="utf-8")
    print(f"  原始行数: {len(df):,}")

    # 按日期分组，取每日最大值（累计计数器的最终值）
    daily = df.groupby("日期").agg(
        合格数=("合格数", "max"),
        不合格数=("不合格数", "max"),
        生产线编号=("生产线编号", "first"),
    ).reset_index()

    # 计算产量和合格率
    daily["产量"] = daily["合格数"] + daily["不合格数"]
    daily["合格率"] = daily["合格数"] / daily["产量"].replace(0, np.nan)
    # 产量为0的天，合格率设为NaN
    daily["合格率"] = daily["合格率"].fillna(0.0)

    print(f"  生产天数: {len(daily)}")
    print(f"  日期范围: 第{daily['日期'].min()}天 ~ 第{daily['日期'].max()}天")
    print(f"  平均日产量: {daily['产量'].mean():.0f}")
    print(f"  平均合格率: {daily['合格率'].mean():.4f}")
    return daily


def load_operator_info() -> pd.DataFrame:
    """加载操作人员信息表。"""
    print("[2/6] 加载操作人员信息表...")
    ops = pd.read_excel(OPERATOR_FILE)
    print(f"  操作人员数: {len(ops)}")
    for _, row in ops.iterrows():
        print(f"    {row['操作人员编号']} → {row['生产线编号']} (工龄{row['工龄']}年)")
    return ops


# ============================================================
# 统计分析
# ============================================================
def run_anova_by_day(daily: pd.DataFrame) -> pd.DataFrame:
    """
    单因素方差分析：不同日期（工作日）对产量和合格率的影响。
    由于只有M301一条生产线，无法做真正的双因素ANOVA。
    替代方案：按日期分组做单因素ANOVA，检验不同日期间产量/合格率是否有显著差异。
    """
    print("[3/6] 方差分析（ANOVA）...")

    results = []

    # --- 产量的ANOVA ---
    # 将每日产量分为前半段和后半段，做组间比较
    n_days = len(daily)
    mid = n_days // 2
    group_a_yield = daily.iloc[:mid]["产量"].values
    group_b_yield = daily.iloc[mid:]["产量"].values

    f_stat_yield, p_val_yield = stats.f_oneway(group_a_yield, group_b_yield)
    results.append({
        "分析项目": "产量（前半段 vs 后半段）",
        "F统计量": round(f_stat_yield, 4),
        "p值": round(p_val_yield, 6),
        "显著性": "显著" if p_val_yield < 0.05 else "不显著",
        "说明": "前半段与后半段工作日的产量均值差异",
    })

    # --- 合格率的ANOVA ---
    group_a_qr = daily.iloc[:mid]["合格率"].values
    group_b_qr = daily.iloc[mid:]["合格率"].values

    f_stat_qr, p_val_qr = stats.f_oneway(group_a_qr, group_b_qr)
    results.append({
        "分析项目": "合格率（前半段 vs 后半段）",
        "F统计量": round(f_stat_qr, 4),
        "p值": round(p_val_qr, 6),
        "显著性": "显著" if p_val_qr < 0.05 else "不显著",
        "说明": "前半段与后半段工作日的合格率均值差异",
    })

    # --- 产量按星期分组ANOVA（如果有足够样本）---
    # 用日期对7取余模拟"星期"分组
    daily_copy = daily.copy()
    daily_copy["周组"] = daily_copy["日期"] % 7
    week_groups_yield = [g["产量"].values for _, g in daily_copy.groupby("周组") if len(g) >= 2]
    if len(week_groups_yield) >= 2:
        f_stat_wk, p_val_wk = stats.f_oneway(*week_groups_yield)
        results.append({
            "分析项目": "产量（按日期模7分组）",
            "F统计量": round(f_stat_wk, 4),
            "p值": round(p_val_wk, 6),
            "显著性": "显著" if p_val_wk < 0.05 else "不显著",
            "说明": "不同周次位置的产量均值差异",
        })

    week_groups_qr = [g["合格率"].values for _, g in daily_copy.groupby("周组") if len(g) >= 2]
    if len(week_groups_qr) >= 2:
        f_stat_wk_qr, p_val_wk_qr = stats.f_oneway(*week_groups_qr)
        results.append({
            "分析项目": "合格率（按日期模7分组）",
            "F统计量": round(f_stat_wk_qr, 4),
            "p值": round(p_val_wk_qr, 6),
            "显著性": "显著" if p_val_wk_qr < 0.05 else "不显著",
            "说明": "不同周次位置的合格率均值差异",
        })

    # --- 相关性分析 ---
    corr_yield_qr, p_corr = stats.pearsonr(daily["产量"], daily["合格率"])
    results.append({
        "分析项目": "产量与合格率Pearson相关",
        "F统计量": round(corr_yield_qr, 4),  # 此处存相关系数
        "p值": round(p_corr, 6),
        "显著性": "显著" if p_corr < 0.05 else "不显著",
        "说明": f"Pearson相关系数={corr_yield_qr:.4f}，产量与合格率的线性关系",
    })

    # --- 生产线编号说明 ---
    results.append({
        "分析项目": "双因素ANOVA说明",
        "F统计量": np.nan,
        "p值": np.nan,
        "显著性": "N/A",
        "说明": "仅有M301一条生产线，无法进行生产线×操作人员的双因素ANOVA。"
                "操作人员信息表中A001对应M301，A002对应M302（无M302数据）。",
    })

    anova_df = pd.DataFrame(results)
    print(f"  共完成 {len([r for r in results if not np.isnan(r['F统计量'])])} 项检验")
    for _, row in anova_df.iterrows():
        if not np.isnan(row["F统计量"]):
            print(f"    {row['分析项目']}: F={row['F统计量']}, p={row['p值']}, {row['显著性']}")
    return anova_df


# ============================================================
# 可视化
# ============================================================
def create_visualizations(daily: pd.DataFrame, ops: pd.DataFrame) -> None:
    """生成4张可视化图表。"""
    print("[4/6] 生成可视化...")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # 设置样式
    sns.set_theme(style="whitegrid")
    setup_chinese_font()

    # --- 图1: 产量按日期箱线图（分组） ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # 将日期分为4组
    daily_copy = daily.copy()
    n = len(daily_copy)
    bins = pd.cut(daily_copy["日期"], bins=4, labels=["第1周", "第2周", "第3-4周", "第5周+"])
    daily_copy["周段"] = bins
    sns.boxplot(data=daily_copy, x="周段", y="产量", palette="Set2", ax=ax)
    ax.set_title("M301生产线各时段产量分布（箱线图）", fontsize=14)
    ax.set_xlabel("时段", fontsize=12)
    ax.set_ylabel("日产量（件）", fontsize=12)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "task4_yield_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ task4_yield_boxplot.png")

    # --- 图2: 合格率按日期分布箱线图 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=daily_copy, x="周段", y="合格率", palette="Set3", ax=ax)
    ax.set_title("M301生产线各时段合格率分布（箱线图）", fontsize=14)
    ax.set_xlabel("时段", fontsize=12)
    ax.set_ylabel("合格率", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2%}"))
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "task4_qualification_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ task4_qualification_boxplot.png")

    # --- 图3: 产量与合格率交互效应图（双Y轴折线图） ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        daily["日期"], daily["产量"],
        "b-o", markersize=5, linewidth=1.5, label="日产量", alpha=0.8
    )
    line2 = ax2.plot(
        daily["日期"], daily["合格率"],
        "r-s", markersize=5, linewidth=1.5, label="合格率", alpha=0.8
    )

    ax1.set_xlabel("日期（天）", fontsize=12)
    ax1.set_ylabel("日产量（件）", fontsize=12, color="blue")
    ax2.set_ylabel("合格率", fontsize=12, color="red")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2%}"))

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    ax1.set_title("M301生产线产量与合格率随时间变化（交互效应图）", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "task4_interaction_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ task4_interaction_plot.png")

    # --- 图4: 相关性热力图 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_cols = ["产量", "合格率", "合格数", "不合格数"]
    corr_matrix = daily[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("产量与合格率相关性热力图", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "task4_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ task4_correlation_heatmap.png")


# ============================================================
# 保存结果
# ============================================================
def save_results(daily: pd.DataFrame, anova_df: pd.DataFrame) -> None:
    """保存分析结果CSV。"""
    print("[5/6] 保存分析结果...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 产量分析汇总
    daily.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ {ANALYSIS_CSV}")

    # ANOVA结果
    anova_df.to_csv(ANOVA_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ {ANOVA_CSV}")


def print_summary(daily: pd.DataFrame, ops: pd.DataFrame) -> None:
    """打印分析摘要。"""
    print("\n[6/6] 分析摘要")
    print("=" * 60)
    print(f"生产线: M301（唯一可用生产线）")
    print(f"操作人员: A001（工龄2年）对应M301")
    print(f"生产天数: {len(daily)} 天")
    print(f"\n--- 产量统计 ---")
    print(f"  总产量: {daily['产量'].sum():,} 件")
    print(f"  日均产量: {daily['产量'].mean():,.0f} 件")
    print(f"  产量标准差: {daily['产量'].std():,.0f}")
    print(f"  最高日产量: {daily['产量'].max():,} 件（第{daily.loc[daily['产量'].idxmax(), '日期']}天）")
    print(f"  最低日产量: {daily['产量'].min():,} 件（第{daily.loc[daily['产量'].idxmin(), '日期']}天）")
    print(f"\n--- 合格率统计 ---")
    print(f"  平均合格率: {daily['合格率'].mean():.4f} ({daily['合格率'].mean()*100:.2f}%)")
    print(f"  合格率标准差: {daily['合格率'].std():.4f}")
    print(f"  最高合格率: {daily['合格率'].max():.4f}（第{daily.loc[daily['合格率'].idxmax(), '日期']}天）")
    print(f"  最低合格率: {daily['合格率'].min():.4f}（第{daily.loc[daily['合格率'].idxmin(), '日期']}天）")

    # 相关性
    corr, p = stats.pearsonr(daily["产量"], daily["合格率"])
    print(f"\n--- 产量与合格率关系 ---")
    print(f"  Pearson相关系数: {corr:.4f}")
    print(f"  p值: {p:.6f}")
    if p < 0.05:
        direction = "正相关" if corr > 0 else "负相关"
        print(f"  结论: 产量与合格率存在显著{direction}（p<0.05）")
    else:
        print(f"  结论: 产量与合格率无显著线性相关（p≥0.05）")

    print(f"\n--- 双因素ANOVA说明 ---")
    print(f"  当前数据仅包含M301一条生产线，操作人员A001。")
    print(f"  无法进行「生产线×操作人员」的双因素方差分析。")
    print(f"  已用单因素ANOVA替代：检验不同时间段产量/合格率的差异。")
    print("=" * 60)


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    print("=" * 60)
    print("任务4 — 产量与合格率关系分析")
    print("=" * 60)

    # 1. 加载数据
    daily = load_production_data()
    ops = load_operator_info()

    # 2. 合并操作人员信息（M301→A001）
    daily["操作人员"] = "A001"  # M301对应A001

    # 3. 方差分析
    anova_df = run_anova_by_day(daily)

    # 4. 可视化
    create_visualizations(daily, ops)

    # 5. 保存结果
    save_results(daily, anova_df)

    # 6. 打印摘要
    print_summary(daily, ops)

    print("\n✓ 所有输出文件:")
    print(f"  {ANALYSIS_CSV}")
    print(f"  {ANOVA_CSV}")
    for png in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {png}")
    print("\n完成！")


if __name__ == "__main__":
    main()
