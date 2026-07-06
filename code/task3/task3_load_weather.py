#!/usr/bin/env python3
"""
气象数据加载与合并脚本

加载附件4的气象信息CSV（GBK编码），解析日期、天气、温度、风力，
与旅客客流数据按 日期+站点 合并，输出合并后的CSV。

用法: python3 code/task3_load_weather.py
"""

import os
from pathlib import Path

import pandas as pd

# 路径配置
WEATHER_FILE = (
    Path(__file__).parent.parent
    / "课程设计资料"
    / "泰迪杯2016年B题_说明及数据"
    / "附件4：车站所属地区气象信息.csv"
)
PASSENGER_FILE = Path(__file__).parent / "output" / "task3_passenger_data.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "task3_merged_data.csv"


def load_weather(filepath: Path) -> pd.DataFrame:
    """
    加载气象CSV，尝试多种编码。

    原始列: 日期, 天气状况, 温度, 风向风力, 车站编码
    输出列: 日期(日期), 车站编码, 天气_白天, 天气_夜间, 最高温, 最低温, 风力_白天, 风力_夜间
    """
    # 尝试多种编码读取
    df = None
    for enc in ("gbk", "gb2312", "gb18030", "latin1"):
        try:
            df = pd.read_csv(filepath, encoding=enc, header=0)
            print(f"  成功使用编码 {enc} 读取气象文件")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df is None:
        raise RuntimeError(f"无法读取气象文件: {filepath}")

    print(f"  原始气象数据: {len(df)} 行, 列名: {list(df.columns)}")

    # 标准化列名（去除BOM和空格）
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    # 统一列名（实际列名可能为 气温/温度, 地区/车站编码 等变体）
    col_rename = {}
    for c in df.columns:
        cl = c.strip()
        if cl in ("气温", "温度"):
            col_rename[c] = "温度"
        elif cl in ("地区", "车站编码", "车站编号"):
            col_rename[c] = "车站编码"
        elif cl in ("风向风力", "风力风向"):
            col_rename[c] = "风向风力"
    if col_rename:
        df = df.rename(columns=col_rename)
        print(f"  列名映射: {col_rename}")

    # 如果列名是乱码，按位置重命名
    if "日期" not in df.columns:
        df.columns = ["日期", "天气状况", "温度", "风向风力", "车站编码"]
        print("  列名已按位置重命名")

    # --- 解析日期: "2015年1月1日" → datetime ---
    df["日期"] = pd.to_datetime(df["日期"], format="%Y年%m月%d日", errors="coerce")
    bad_dates = df["日期"].isna().sum()
    if bad_dates > 0:
        print(f"  警告: {bad_dates} 行日期解析失败，已丢弃")
        df = df.dropna(subset=["日期"])

    # --- 解析天气: "晴/晴" → 天气_白天, 天气_夜间 ---
    weather_split = df["天气状况"].astype(str).str.split("/", n=1, expand=True)
    df["天气_白天"] = weather_split[0]
    df["天气_夜间"] = weather_split[1] if weather_split.shape[1] > 1 else ""

    # --- 解析温度: "4℃/-3℃" → 最高温, 最低温 ---
    temp_split = df["温度"].astype(str).str.replace("℃", "", regex=False).str.split("/", n=1, expand=True)
    df["最高温"] = pd.to_numeric(temp_split[0], errors="coerce")
    df["最低温"] = pd.to_numeric(temp_split[1], errors="coerce") if temp_split.shape[1] > 1 else None

    # --- 解析风力: "北风3级/北风3级" → 风力_白天, 风力_夜间 ---
    wind_split = df["风向风力"].astype(str).str.split("/", n=1, expand=True)
    df["风力_白天"] = wind_split[0]
    df["风力_夜间"] = wind_split[1] if wind_split.shape[1] > 1 else ""

    # 保留需要的列
    keep_cols = ["日期", "车站编码", "天气_白天", "天气_夜间", "最高温", "最低温", "风力_白天", "风力_夜间"]
    df = df[keep_cols]

    print(f"  解析后气象数据: {len(df)} 行, {df['车站编码'].nunique()} 个站点")
    print(f"  日期范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

    return df


def load_passenger(filepath: Path) -> pd.DataFrame:
    """
    加载旅客客流CSV。

    列: 日期, 车次, 始发站, 终到站, 上车站, 下车站, 客流人数
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    print(f"  旅客数据: {len(df)} 行")
    print(f"  日期范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")
    print(f"  上车站: {df['上车站'].nunique()} 个")
    return df


def merge_data(passenger_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 (日期, 上车站) 左连接合并旅客数据与气象数据。
    """
    # 重命名气象的车站编码以便合并键对齐
    weather_rename = weather_df.rename(columns={"车站编码": "上车站"})

    print(f"\n合并前: 旅客 {len(passenger_df)} 行, 气象 {len(weather_rename)} 行")

    merged = passenger_df.merge(
        weather_rename,
        on=["日期", "上车站"],
        how="left",
    )

    # 统计匹配情况
    weather_cols = ["天气_白天", "天气_夜间", "最高温", "最低温", "风力_白天", "风力_夜间"]
    matched = merged[weather_cols[0]].notna().sum()
    print(f"合并后: {len(merged)} 行")
    print(f"气象匹配率: {matched}/{len(merged)} ({matched / len(merged) * 100:.1f}%)")

    # 列出未匹配的站点（可能气象数据不覆盖所有站点）
    if matched < len(merged):
        unmatched = merged[merged[weather_cols[0]].isna()]["上车站"].unique()
        print(f"未匹配站点（气象无覆盖）: {sorted(unmatched)[:20]}")

    return merged


def main() -> None:
    """主流程: 加载 → 解析 → 合并 → 输出"""
    # 检查输入文件
    if not WEATHER_FILE.exists():
        print(f"错误: 气象文件不存在 {WEATHER_FILE}")
        return
    if not PASSENGER_FILE.exists():
        print(f"错误: 旅客数据文件不存在 {PASSENGER_FILE}")
        print("  请先运行 python3 code/task3_load_data.py 生成旅客数据")
        return

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("1. 加载气象数据...")
    weather_df = load_weather(WEATHER_FILE)

    print("\n2. 加载旅客数据...")
    passenger_df = load_passenger(PASSENGER_FILE)

    # 合并
    print("\n3. 合并数据...")
    merged_df = merge_data(passenger_df, weather_df)

    # 输出
    merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n完成! 合并数据已保存至: {OUTPUT_FILE}")
    print(f"总行数: {len(merged_df)}, 列数: {len(merged_df.columns)}")
    print(f"列名: {list(merged_df.columns)}")


if __name__ == "__main__":
    main()
