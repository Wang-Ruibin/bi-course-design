#!/usr/bin/env python3
"""
旅客列车梯形密度表数据加载脚本

从440个xls文件中解析旅客列车梯形密度表数据，
提取日期、车次、始发站、终到站、上车站、下车站、客流人数，
输出到CSV文件供后续分析使用。

支持两种xls格式：
  1. 标准xls（xlrd解析）
  2. HTML格式xls（WPS Office导出，pd.read_html解析）

用法: python3 code/task3_load_data.py
"""

import os
import re
import glob
import pandas as pd
import xlrd
from pathlib import Path

# 数据根目录
DATA_DIR = Path(__file__).parent.parent / "课程设计资料" / "泰迪杯2016年B题_说明及数据" / "附件1：旅客列车梯形密度表"
# 输出路径
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "task3_passenger_data.csv"


def parse_date_from_filename(filename):
    """
    从文件名解析日期，如 "20151001.xls" → "2015-10-01"
    """
    basename = Path(filename).stem  # 去掉扩展名
    if len(basename) == 8 and basename.isdigit():
        return f"{basename[:4]}-{basename[4:6]}-{basename[6:8]}"
    return None


def extract_records_from_grid(grid, date_str):
    """
    从二维列表（grid）中提取客流记录

    grid格式:
      Row 0: 标题行（忽略）
      Row 1: 日期行（忽略）
      Row 2: 车次信息行
      Row 3: 站点编码行（上车站标题 + 空 + 站点编码...）
      Row 4: 到点行（下车站标题 + 开/到点 + 时间...）
      Row 5+: 数据矩阵

    每条记录: (日期, 车次, 始发站, 终到站, 上车站, 下车站, 客流人数)
    """
    if len(grid) < 6:
        return None, "行数不足"

    # 解析Row 2：车次信息
    row2 = str(grid[2][0]) if grid[2] else ""
    m = re.match(r'(\S+)\s+(\S+?)—(\S+?)(?:-\d+)?\s', row2)
    if not m:
        return None, f"无法解析车次信息: {row2[:60]}"
    train_no, origin, dest = m.group(1), m.group(2), m.group(3)

    # 解析Row 3：站点编码（从列2开始）
    stations = []
    row3 = grid[3]
    for c in range(2, len(row3)):
        val = row3[c]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            break
        val_str = str(val).strip()
        if not val_str or '合计' in val_str:
            break
        stations.append(val_str)

    if not stations:
        return None, "无法提取站点编码"

    num_stations = len(stations)

    # 遍历数据矩阵（Row 5开始）
    records = []
    for r in range(5, len(grid)):
        row = grid[r]
        if not row or len(row) < 3:
            continue

        # 检查是否到达汇总行或数据源行
        first_cell = str(row[0]).strip() if row[0] else ""
        if '合计' in first_cell or '数据源' in first_cell:
            break

        # 上车站（第一列）
        boarding_station = first_cell
        if not boarding_station:
            continue

        # 遍历该行的客流数据（列2开始，对应站点编码）
        for c in range(2, 2 + num_stations):
            if c >= len(row):
                break
            val = row[c]
            # 跳过空值
            if val is None or val == '' or (isinstance(val, float) and pd.isna(val)):
                continue
            try:
                count = float(val)
            except (ValueError, TypeError):
                continue
            if count <= 0:
                continue

            alighting_station = stations[c - 2]
            # 跳过自环（上车站 == 下车站）
            if boarding_station == alighting_station:
                continue

            records.append((
                date_str,
                train_no,
                origin,
                dest,
                boarding_station,
                alighting_station,
                int(count),
            ))

    return records, None


def parse_single_file(filepath):
    """
    解析单个xls文件，返回记录列表

    自动识别两种格式：
      1. 标准xls → xlrd
      2. HTML格式xls → pd.read_html
    """
    date_str = parse_date_from_filename(filepath.name)
    if not date_str:
        return None, "无法从文件名解析日期"

    # 尝试方式1：标准xls（xlrd）
    try:
        wb = xlrd.open_workbook(str(filepath))
        ws = wb.sheet_by_index(0)
        # 将工作表转为二维列表
        grid = []
        for r in range(ws.nrows):
            grid.append([ws.cell_value(r, c) for c in range(ws.ncols)])
        return extract_records_from_grid(grid, date_str)
    except xlrd.XLRDError:
        pass  # 非标准xls，尝试HTML格式

    # 尝试方式2：HTML格式xls（WPS Office导出）
    try:
        dfs = pd.read_html(str(filepath), encoding='utf-8')
        if not dfs:
            return None, "HTML文件中未找到表格"
        df = dfs[0]
        # 转为二维列表（NaN替换为None）
        grid = df.where(df.notna(), other=None).values.tolist()
        return extract_records_from_grid(grid, date_str)
    except Exception as e:
        return None, f"两种格式均无法解析: {e}"


def main():
    """主函数：遍历所有xls文件，解析并汇总数据"""
    # 检查数据目录
    if not DATA_DIR.exists():
        print(f"错误: 数据目录不存在 {DATA_DIR}")
        return

    # 收集所有xls文件
    xls_files = sorted(glob.glob(str(DATA_DIR / "**" / "*.xls"), recursive=True))
    total_files = len(xls_files)
    print(f"找到 {total_files} 个xls文件")

    if total_files == 0:
        print("错误: 未找到任何xls文件")
        return

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 统计格式分布
    xlrd_count = 0
    html_count = 0

    # 逐文件解析
    all_records = []
    errors = []
    for i, filepath in enumerate(xls_files, 1):
        fp = Path(filepath)

        # 先判断格式
        try:
            xlrd.open_workbook(str(fp))
            xlrd_count += 1
        except xlrd.XLRDError:
            html_count += 1

        records, err = parse_single_file(fp)
        if err:
            errors.append((filepath, err))
        elif records:
            all_records.extend(records)

        # 每50个文件打印进度
        if i % 50 == 0 or i == total_files:
            print(f"  已处理 {i}/{total_files} 个文件，累计 {len(all_records)} 条记录")

    # 打印格式统计
    print(f"\n格式分布: 标准xls={xlrd_count}, HTML格式xls={html_count}")

    # 打印错误汇总
    if errors:
        print(f"警告: {len(errors)} 个文件解析失败:")
        for fp, err in errors[:10]:
            print(f"  - {Path(fp).name}: {err}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")

    # 输出CSV
    if not all_records:
        print("错误: 未解析到任何数据")
        return

    df = pd.DataFrame(
        all_records,
        columns=["日期", "车次", "始发站", "终到站", "上车站", "下车站", "客流人数"],
    )
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n完成! 共 {len(df)} 条记录")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"日期范围: {df['日期'].min()} ~ {df['日期'].max()}")
    print(f"车次: {df['车次'].unique()}")
    print(f"站点数: 上车 {df['上车站'].nunique()}, 下车 {df['下车站'].nunique()}")


if __name__ == "__main__":
    main()
