#!/usr/bin/env python3
"""
任务3：多维客流分析与可视化
对合并后的客运数据进行探索性数据分析（EDA），生成至少8张可视化图表。

数据来源：task3_merged_data.csv（K11次列车，2015-01 至 2016-03）
输出目录：code/output/task3_eda/
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头渲染，不依赖显示器
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# 0. 全局设置：中文字体 + 图表风格
# ---------------------------------------------------------------------------

# 注册系统中所有 CJK 字体，确保 matplotlib 能找到
_FONT_CANDIDATES = [
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "Noto Sans CJK SC", "Noto Sans CJK JP",
    "SimHei", "Microsoft YaHei", "AR PL UMing CN",
]

# 刷新 matplotlib 字体缓存（确保新安装的字体被识别）
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

# 从系统字体列表中找出实际存在的 CJK 字体名
_available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_chosen = [f for f in _FONT_CANDIDATES if f in _available_fonts]

if _chosen:
    plt.rcParams["font.sans-serif"] = _chosen + ["DejaVu Sans"]
    print(f"[FONT] 使用中文字体: {_chosen[0]}")
else:
    # 最后手段：直接指定字体文件路径
    _font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in _font_paths:
        if os.path.exists(fp):
            matplotlib.font_manager.fontManager.addfont(fp)
            prop = matplotlib.font_manager.FontProperties(fname=fp)
            plt.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
            print(f"[FONT] 从文件加载中文字体: {fp}")
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("[WARN] 未找到中文字体，图表中文将显示为方块")

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "task3_eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据路径
DATA_PATH = Path(__file__).resolve().parent / "output" / "task3_merged_data.csv"


# ---------------------------------------------------------------------------
# 1. 加载与清洗数据
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """加载合并数据并进行基础清洗。"""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # 删除日期为空的行（约占0.3%）
    df = df.dropna(subset=["日期"]).copy()

    # 解析日期
    df["日期"] = pd.to_datetime(df["日期"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["日期"]).copy()

    # 提取时间维度
    df["年月"] = df["日期"].dt.to_period("M")
    df["月份"] = df["日期"].dt.month
    df["年份"] = df["日期"].dt.year
    df["星期"] = df["日期"].dt.dayofweek  # 0=周一
    df["季度"] = df["日期"].dt.quarter

    # 构造区间字段
    df["区间"] = df["上车站"] + " → " + df["下车站"]

    # 构造路线字段（始发站-终到站）
    df["路线"] = df["始发站"] + " → " + df["终到站"]

    # 客流人数转数值
    df["客流人数"] = pd.to_numeric(df["客流人数"], errors="coerce").fillna(0).astype(int)

    # 天气数据：标记有无
    df["有天气数据"] = df["天气_白天"].notna()

    print(f"[INFO] 加载数据完成：{len(df):,} 行，"
          f"日期范围 {df['日期'].min().date()} ~ {df['日期'].max().date()}")
    print(f"[INFO] 总客流人数：{df['客流人数'].sum():,}")
    print(f"[INFO] 有天气数据的行数：{df['有天气数据'].sum():,} "
          f"({df['有天气数据'].mean()*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# 2. 可视化函数
# ---------------------------------------------------------------------------

def plot_01_flow_by_route(df: pd.DataFrame) -> None:
    """图1：按路线（始发站→终到站）统计客流总量柱状图。
    注：数据中仅有K11次列车，故改为按路线维度展示。"""
    route_flow = (df.groupby("路线")["客流人数"]
                    .sum()
                    .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(10, max(5, len(route_flow) * 0.5)))
    bars = ax.barh(route_flow.index, route_flow.values, color=sns.color_palette("viridis", len(route_flow)))

    # 在柱状图末端标注数值
    for bar, val in zip(bars, route_flow.values):
        ax.text(val + route_flow.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=10)

    ax.set_xlabel("客流总人数", fontsize=12)
    ax.set_title("图1  K11次列车各路线客流总量", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_flow_by_route.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 01_flow_by_route.png")


def plot_02_daily_trend(df: pd.DataFrame) -> None:
    """图2：每日客流趋势折线图（含7日移动平均）。"""
    daily = (df.groupby("日期")["客流人数"]
               .sum()
               .sort_index())

    # 7日移动平均
    ma7 = daily.rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(daily.index, daily.values, alpha=0.35, linewidth=0.8,
            color="#4c72b0", label="每日客流")
    ax.plot(ma7.index, ma7.values, linewidth=2, color="#c44e52",
            label="7日移动平均")

    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("客流人数", fontsize=12)
    ax.set_title("图2  每日客流趋势（含7日移动平均）", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_daily_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 02_daily_trend.png")


def plot_03_flow_by_station(df: pd.DataFrame) -> None:
    """图3：上车站客流排名（水平柱状图，Top 20）。"""
    station_flow = (df.groupby("上车站")["客流人数"]
                      .sum()
                      .sort_values(ascending=False)
                      .head(20)
                      .sort_values(ascending=True))  # 画图时从下到上递增

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("YlOrRd_r", len(station_flow))
    bars = ax.barh(station_flow.index, station_flow.values, color=palette)

    for bar, val in zip(bars, station_flow.values):
        ax.text(val + station_flow.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=9)

    ax.set_xlabel("客流总人数", fontsize=12)
    ax.set_title("图3  上车站客流排名 Top 20", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_flow_by_station.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 03_flow_by_station.png")


def plot_04_flow_by_interval(df: pd.DataFrame) -> None:
    """图4：区间（上车站→下车站）客流排名 Top 20。"""
    interval_flow = (df.groupby("区间")["客流人数"]
                       .sum()
                       .sort_values(ascending=False)
                       .head(20)
                       .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(11, 8))
    palette = sns.color_palette("coolwarm_r", len(interval_flow))
    bars = ax.barh(interval_flow.index, interval_flow.values, color=palette)

    for bar, val in zip(bars, interval_flow.values):
        ax.text(val + interval_flow.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=9)

    ax.set_xlabel("客流总人数", fontsize=12)
    ax.set_title("图4  区间客流排名 Top 20（上车站→下车站）", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_flow_by_interval.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 04_flow_by_interval.png")


def plot_05_monthly_trend(df: pd.DataFrame) -> None:
    """图5：月度客流趋势折线图。"""
    monthly = (df.groupby("年月")["客流人数"]
                 .sum()
                 .sort_index())

    # 转为字符串用于x轴标签
    x_labels = [str(p) for p in monthly.index]
    x_pos = range(len(x_labels))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x_pos, monthly.values, marker="o", markersize=5, linewidth=2,
            color="#55a868", markerfacecolor="#ffffff", markeredgewidth=1.5,
            markeredgecolor="#55a868")

    # 标注峰值
    peak_idx = monthly.values.argmax()
    ax.annotate(f"{monthly.values[peak_idx]:,.0f}",
                xy=(peak_idx, monthly.values[peak_idx]),
                xytext=(0, 15), textcoords="offset points",
                fontsize=9, ha="center", color="#c44e52",
                arrowprops=dict(arrowstyle="->", color="#c44e52"))

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("客流总人数", fontsize=12)
    ax.set_title("图5  月度客流趋势（2015-01 ~ 2016-03）", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_monthly_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 05_monthly_trend.png")


def plot_06_weather_impact(df: pd.DataFrame) -> None:
    """图6：不同天气条件下日均客流对比。"""
    weather_df = df[df["有天气数据"]].copy()
    if weather_df.empty:
        print("[SKIP] 06_weather_impact.png — 无天气数据")
        return

    # 按天气类型统计每日平均客流
    weather_daily = (weather_df.groupby(["天气_白天", "日期"])["客流人数"]
                               .sum()
                               .reset_index())
    weather_avg = (weather_daily.groupby("天气_白天")["客流人数"]
                                .agg(["mean", "std", "count"])
                                .sort_values("mean", ascending=False))

    # 仅保留出现次数≥3天的天气类型，避免小样本误导
    weather_avg = weather_avg[weather_avg["count"] >= 3]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(weather_avg))
    bars = ax.bar(weather_avg.index, weather_avg["mean"],
                  yerr=weather_avg["std"] / np.sqrt(weather_avg["count"]),
                  capsize=4, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, val in zip(bars, weather_avg["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("白天天气类型", fontsize=12)
    ax.set_ylabel("日均客流人数（±标准误）", fontsize=12)
    ax.set_title("图6  不同天气条件下日均客流对比", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_weather_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 06_weather_impact.png")


def plot_07_station_heatmap(df: pd.DataFrame) -> None:
    """图7：主要站点间客流热力图（上车站 × 下车站）。"""
    # 选取客流量最大的前10个上车站和前10个下车站
    top_boarding = (df.groupby("上车站")["客流人数"].sum()
                      .nlargest(10).index.tolist())
    top_alighting = (df.groupby("下车站")["客流人数"].sum()
                       .nlargest(10).index.tolist())

    # 构建矩阵
    pivot = (df[df["上车站"].isin(top_boarding) & df["下车站"].isin(top_alighting)]
             .groupby(["上车站", "下车站"])["客流人数"]
             .sum()
             .unstack(fill_value=0))

    # 确保行列顺序一致
    pivot = pivot.reindex(index=top_boarding, columns=top_alighting, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(pivot, annot=True, fmt=",", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "客流总人数"}, ax=ax)

    ax.set_xlabel("下车站", fontsize=12)
    ax.set_ylabel("上车站", fontsize=12)
    ax.set_title("图7  主要站点间客流热力图（Top 10 × Top 10）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_station_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 07_station_heatmap.png")


def _get_season(month: int) -> str:
    """月份转季节。"""
    if month in (3, 4, 5):
        return "春季"
    elif month in (6, 7, 8):
        return "夏季"
    elif month in (9, 10, 11):
        return "秋季"
    else:
        return "冬季"


def plot_08_seasonal_boxplot(df: pd.DataFrame) -> None:
    """图8：季节性日客流箱线图。"""
    daily_season = (df.groupby(["日期", "月份"])["客流人数"]
                      .sum()
                      .reset_index())
    daily_season["季节"] = daily_season["月份"].apply(_get_season)

    season_order = ["春季", "夏季", "秋季", "冬季"]
    # 仅保留有数据的季节
    present_seasons = [s for s in season_order if s in daily_season["季节"].values]

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"春季": "#8db596", "夏季": "#f4a259",
               "秋季": "#e76f51", "冬季": "#457b9d"}

    sns.boxplot(data=daily_season, x="季节", y="客流人数",
                order=present_seasons,
                palette={s: palette[s] for s in present_seasons},
                width=0.5, fliersize=3, ax=ax)
    sns.stripplot(data=daily_season, x="季节", y="客流人数",
                  order=present_seasons, color="black", alpha=0.15,
                  size=2, jitter=True, ax=ax)

    ax.set_xlabel("季节", fontsize=12)
    ax.set_ylabel("日客流人数", fontsize=12)
    ax.set_title("图8  各季节日客流分布箱线图", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_seasonal_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 08_seasonal_boxplot.png")


def plot_09_weekday_pattern(df: pd.DataFrame) -> None:
    """图9：星期客流分布（星期一~星期日的日均客流）。"""
    daily_weekday = (df.groupby(["日期", "星期"])["客流人数"]
                       .sum()
                       .reset_index())
    weekday_avg = (daily_weekday.groupby("星期")["客流人数"]
                                 .mean()
                                 .reindex(range(7)))

    labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    colors = ["#4c72b0"] * 5 + ["#dd8452"] * 2  # 工作日/周末区分色

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, weekday_avg.values, color=colors, edgecolor="gray",
                  linewidth=0.5)

    for bar, val in zip(bars, weekday_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("星期", fontsize=12)
    ax.set_ylabel("日均客流人数", fontsize=12)
    ax.set_title("图9  星期客流分布（工作日 vs 周末）", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 添加图例说明颜色含义
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4c72b0", label="工作日"),
                       Patch(facecolor="#dd8452", label="周末")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_weekday_pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 09_weekday_pattern.png")


def plot_10_top_destination(df: pd.DataFrame) -> None:
    """图10：下车站客流排名 Top 15（与上车站对比的水平堆叠柱状图）。"""
    top_n = 15
    boarding = (df.groupby("上车站")["客流人数"].sum()
                  .nlargest(top_n))
    alighting = (df.groupby("下车站")["客流人数"].sum()
                   .nlargest(top_n))

    # 取并集站点
    all_stations = sorted(set(boarding.index) | set(alighting.index),
                          key=lambda s: boarding.get(s, 0) + alighting.get(s, 0),
                          reverse=True)[:top_n]

    board_vals = [boarding.get(s, 0) for s in all_stations]
    alight_vals = [alighting.get(s, 0) for s in all_stations]

    y_pos = np.arange(len(all_stations))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y_pos, board_vals, height=0.4, label="上车客流", color="#4c72b0")
    ax.barh(y_pos + 0.4, alight_vals, height=0.4, label="下车客流", color="#dd8452")

    ax.set_yticks(y_pos + 0.2)
    ax.set_yticklabels(all_stations)
    ax.invert_yaxis()
    ax.set_xlabel("客流总人数", fontsize=12)
    ax.set_title("图10  主要站点上下车客流对比 Top 15", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_top_destination.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 10_top_destination.png")


# ---------------------------------------------------------------------------
# 3. 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    """执行全部EDA可视化。"""
    print("=" * 60)
    print("任务3：多维客流分析 — EDA可视化")
    print("=" * 60)

    # 加载数据
    df = load_data()

    # 依次生成图表
    plot_01_flow_by_route(df)
    plot_02_daily_trend(df)
    plot_03_flow_by_station(df)
    plot_04_flow_by_interval(df)
    plot_05_monthly_trend(df)
    plot_06_weather_impact(df)
    plot_07_station_heatmap(df)
    plot_08_seasonal_boxplot(df)
    plot_09_weekday_pattern(df)
    plot_10_top_destination(df)

    # 汇总
    pngs = sorted(OUTPUT_DIR.glob("*.png"))
    print("\n" + "=" * 60)
    print(f"完成！共生成 {len(pngs)} 张图表：")
    for p in pngs:
        print(f"  {p.name}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
