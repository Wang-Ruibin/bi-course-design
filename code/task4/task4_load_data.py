#!/usr/bin/env python3
"""
task4_load_data.py — 任务4：生产线故障分析 — 数据加载与清洗

功能：
1. 加载附件1目录下所有M*.csv文件（GBK/GB2312/GB18030编码）
2. 数据清洗：去重、填缺失值、去全空行
3. 特征工程：故障标记、时间特征
4. 输出清洗后的CSV
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_ROOT
    / "课程设计资料"
    / "泰迪杯2024年A题_说明及数据"
    / "A题-示例数据"
    / "附件1"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "task4_cleaned_data.csv"

# 故障列名关键词（编码规则：千位=装置类型，后三位=故障码）
FAULT_COLS_KEYWORDS = [
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

# 编码尝试顺序
ENCODINGS = ["gbk", "gb2312", "gb18030", "latin1"]


def _read_csv_safe(filepath: Path) -> pd.DataFrame:
    """尝试多种编码读取CSV，返回DataFrame或抛出异常。"""
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"  ✓ 成功以 {enc} 编码读取: {filepath.name} ({len(df)} 行)")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"  ✗ 读取失败 {filepath.name}: {e}")
            raise
    raise ValueError(f"所有编码均无法读取: {filepath}")


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    """加载data_dir下所有M*.csv文件并合并。"""
    csv_files = sorted(glob.glob(str(data_dir / "M*.csv")))
    if not csv_files:
        print(f"⚠ 未找到任何M*.csv文件，目录: {data_dir}")
        sys.exit(1)

    print(f"找到 {len(csv_files)} 个CSV文件:")
    frames: list[pd.DataFrame] = []
    for f in csv_files:
        fp = Path(f)
        df = _read_csv_safe(fp)
        # 确保生产线编号列存在
        if "生产线编号" not in df.columns:
            df["生产线编号"] = fp.stem  # 用文件名作为生产线编号
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n合并后总行数: {len(combined):,}")
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：
    1. 识别传感器列（除日期、时间、生产线编号外的数值列）
    2. 删除传感器列全部为NaN的行
    3. 数值列NaN填充为0
    4. 去除完全重复行
    """
    initial_rows = len(df)

    # 定义元数据列
    meta_cols = ["日期", "时间", "生产线编号"]
    # 传感器/数值列 = 除元数据外的所有列
    sensor_cols = [c for c in df.columns if c not in meta_cols]

    # 转换数值列为numeric（无法转换的变NaN）
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除传感器列全为NaN的行
    df = df.dropna(subset=sensor_cols, how="all")
    dropped_na = initial_rows - len(df)
    print(f"删除传感器全空行: {dropped_na} 行")

    # 填充NaN为0
    df[sensor_cols] = df[sensor_cols].fillna(0)

    # 去除重复行
    before_dedup = len(df)
    df = df.drop_duplicates()
    dropped_dup = before_dedup - len(df)
    print(f"删除重复行: {dropped_dup} 行")

    print(f"清洗后行数: {len(df):,} (原 {initial_rows:,})")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：
    1. 故障标记：任一故障列 > 0 则为1，否则为0
    2. hour_of_day：直接取时间列
    3. is_daytime：6-18点为白天(1)，其余为夜间(0)
    """
    # 故障标记
    existing_fault_cols = [c for c in FAULT_COLS_KEYWORDS if c in df.columns]
    if existing_fault_cols:
        df["故障标记"] = (
            df[existing_fault_cols].gt(0).any(axis=1).astype(int)
        )
        fault_count = df["故障标记"].sum()
        print(f"故障样本数: {fault_count:,} / {len(df):,} "
              f"({fault_count / len(df) * 100:.2f}%)")
    else:
        print("⚠ 未找到故障列，跳过故障标记")
        df["故障标记"] = 0

    # 时间特征
    if "时间" in df.columns:
        df["hour_of_day"] = df["时间"].astype(int)
        df["is_daytime"] = df["hour_of_day"].between(6, 18).astype(int)
        print(f"白天样本: {df['is_daytime'].sum():,} / {len(df):,}")

    return df


def main() -> None:
    print("=" * 60)
    print("任务4 — 生产线故障分析 — 数据加载与清洗")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载CSV文件...")
    df = load_all_csvs(DATA_DIR)

    # 2. 数据清洗
    print("\n[2/4] 数据清洗...")
    df = clean_data(df)

    # 3. 特征工程
    print("\n[3/4] 特征工程...")
    df = engineer_features(df)

    # 4. 保存
    print("\n[4/4] 保存清洗数据...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✓ 已保存: {OUTPUT_FILE}")
    print(f"  行数: {len(df):,}, 列数: {len(df.columns)}")
    print(f"  列名: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
