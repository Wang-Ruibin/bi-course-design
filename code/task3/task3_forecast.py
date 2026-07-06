"""
任务3: 客流量预测与D02-D19列车配置优化
1. 使用最佳模型预测未来2周客流量
2. 基于预测客流优化D02-D19列车配置
3. 生成可视化图表
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
VIZ_DIR = os.path.join(OUTPUT_DIR, 'task3_forecast_viz')
os.makedirs(VIZ_DIR, exist_ok=True)

# D02-D19列车列表
TRAIN_LIST = [f'D{str(i).zfill(2)}' for i in range(2, 20)]


def load_and_aggregate(path: str) -> pd.DataFrame:
    """加载数据并按日聚合客流量"""
    print("=" * 60)
    print("步骤1: 加载并聚合数据")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"原始数据形状: {df.shape}")

    # 解析日期
    df['日期'] = pd.to_datetime(df['日期'])

    # 提取每日天气信息
    weather_cols = ['天气_白天', '天气_夜间', '最高温', '最低温', '风力_白天', '风力_夜间']
    weather_df = df.dropna(subset=['天气_白天']).drop_duplicates('日期')[['日期'] + weather_cols]

    # 按日聚合客流
    daily = df.groupby('日期').agg(
        客流人数=('客流人数', 'sum'),
        车次数=('车次', 'nunique'),
        上车站数=('上车站', 'nunique'),
        下车站数=('下车站', 'nunique'),
    ).reset_index()

    # 合并天气
    daily = daily.merge(weather_df, on='日期', how='left')
    daily = daily.sort_values('日期').reset_index(drop=True)

    print(f"聚合后数据形状: {daily.shape}")
    print(f"日期范围: {daily['日期'].min().date()} ~ {daily['日期'].max().date()}")
    return daily


def engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """特征工程：时间特征、滞后特征、天气特征"""
    print("\n" + "=" * 60)
    print("步骤2: 特征工程（历史数据）")
    print("=" * 60)

    df = daily.copy()

    # --- 时间特征 ---
    df['day_of_week'] = df['日期'].dt.dayofweek
    df['month'] = df['日期'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_year'] = df['日期'].dt.dayofyear
    df['day_of_month'] = df['日期'].dt.day

    # --- 滞后特征 ---
    df['rolling_7d_mean'] = df['客流人数'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_7d_std'] = df['客流人数'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    df['lag_1'] = df['客流人数'].shift(1)
    df['lag_7'] = df['客流人数'].shift(7)

    # --- 天气编码（用数值填充缺失） ---
    df['天气_白天_enc'] = 0
    df['天气_夜间_enc'] = 0
    df['风力_白天_enc'] = 0
    df['风力_夜间_enc'] = 0

    # 温度填充
    df['最高温'] = df['最高温'].interpolate(method='linear').fillna(df['最高温'].median())
    df['最低温'] = df['最低温'].interpolate(method='linear').fillna(df['最低温'].median())

    # 删除因滞后特征产生的NaN行
    df = df.dropna(subset=['rolling_7d_mean', 'lag_7']).reset_index(drop=True)

    print(f"特征工程后数据形状: {df.shape}")
    return df


def load_station_info(path: str) -> pd.DataFrame:
    """加载站点间客流信息，用于优化停站方案"""
    print("\n" + "=" * 60)
    print("步骤3: 加载站点客流信息")
    print("=" * 60)

    df = pd.read_csv(path)
    df['日期'] = pd.to_datetime(df['日期'])

    # 按站点对聚合客流
    station_flow = df.groupby(['上车站', '下车站']).agg(
        总客流=('客流人数', 'sum'),
        平均日客流=('客流人数', 'mean'),
    ).reset_index()

    # 按上车站聚合
    boarding_flow = df.groupby('上车站').agg(
        总上车客流=('客流人数', 'sum'),
        平均日上车客流=('客流人数', 'mean'),
    ).reset_index().sort_values('总上车客流', ascending=False)

    # 按下车站聚合
    alighting_flow = df.groupby('下车站').agg(
        总下车客流=('客流人数', 'sum'),
        平均日下车客流=('客流人数', 'mean'),
    ).reset_index().sort_values('总下车客流', ascending=False)

    print(f"站点对数量: {len(station_flow)}")
    print(f"上车站数量: {len(boarding_flow)}")
    print(f"\nTop 10 上车站:")
    print(boarding_flow.head(10).to_string(index=False))

    return station_flow, boarding_flow, alighting_flow


def forecast_future(model, daily_df: pd.DataFrame, n_days: int = 14) -> pd.DataFrame:
    """
    使用迭代方式预测未来n天的客流量
    每天的预测值会作为后续天的滞后特征
    """
    print("\n" + "=" * 60)
    print(f"步骤4: 预测未来{n_days}天客流量")
    print("=" * 60)

    # 获取最后7天的客流数据（用于初始滞后特征）
    recent_flows = daily_df['客流人数'].values[-7:].tolist()
    recent_7d_mean = np.mean(recent_flows)
    recent_7d_std = np.std(recent_flows)

    # 获取最近的天气和运营统计（用于填充未来特征）
    last_row = daily_df.iloc[-1]
    median_temp_high = daily_df['最高温'].median()
    median_temp_low = daily_df['最低温'].median()
    median_trains = daily_df['车次数'].median()
    median_stations = daily_df['上车站数'].median()

    # 3月历史同期数据（用于季节性调整）
    march_data = daily_df[daily_df['month'] == 3]
    march_avg_by_dow = march_data.groupby('day_of_week')['客流人数'].mean().to_dict()

    # 生成未来日期
    last_date = daily_df['日期'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')

    predictions = []
    for i, date in enumerate(future_dates):
        # 时间特征
        dow = date.dayofweek
        month = date.month
        is_weekend = int(dow >= 5)
        doy = date.dayofyear
        dom = date.day

        # 滞后特征（使用实际数据 + 之前的预测值）
        lag_1 = recent_flows[-1]
        lag_7 = recent_flows[-7] if len(recent_flows) >= 7 else recent_flows[0]
        rolling_mean = np.mean(recent_flows[-7:])
        rolling_std = np.std(recent_flows[-7:]) if len(recent_flows) >= 2 else 0

        # 天气特征（3月下旬-4月初的典型天气）
        # 武汉3月底4月初: 最高温约15-22°C, 最低温约8-14°C
        temp_high = 18 + 2 * np.sin(doy / 365 * 2 * np.pi)  # 季节性温度
        temp_low = temp_high - 8

        # 构建特征向量（与训练时一致）
        features = np.array([[
            dow, month, is_weekend, doy, dom,
            rolling_mean, rolling_std, lag_1, lag_7,
            temp_high, temp_low,
            0, 0,  # 天气编码
            0, 0,  # 风力编码
            median_trains, median_stations, median_stations,  # 运营统计
        ]])

        # 预测
        pred = model.predict(features)[0]

        # 使用历史同期数据进行微调（加权平均）
        historical_ref = march_avg_by_dow.get(dow, recent_7d_mean)
        # 70%模型预测 + 30%历史同期参考
        pred_adjusted = 0.7 * pred + 0.3 * historical_ref

        # 确保预测值合理（不低于历史最低的50%，不高于最高的150%）
        min_val = daily_df['客流人数'].min() * 0.5
        max_val = daily_df['客流人数'].max() * 1.5
        pred_adjusted = np.clip(pred_adjusted, min_val, max_val)

        predictions.append({
            '日期': date,
            '预测客流': int(round(pred_adjusted)),
        })

        # 更新滞后序列
        recent_flows.append(pred_adjusted)
        if len(recent_flows) > 14:  # 保留最近14天
            recent_flows = recent_flows[-14:]

    forecast_df = pd.DataFrame(predictions)
    forecast_df['日期'] = forecast_df['日期'].dt.strftime('%Y-%m-%d')

    print(f"\n预测结果:")
    print(forecast_df.to_string(index=False))
    print(f"\n预测客流统计:")
    print(f"  均值: {forecast_df['预测客流'].mean():.0f}")
    print(f"  最小: {forecast_df['预测客流'].min()}")
    print(f"  最大: {forecast_df['预测客流'].max()}")

    # 保存
    forecast_path = os.path.join(OUTPUT_DIR, 'task3_forecast.csv')
    forecast_df.to_csv(forecast_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存: {forecast_path}")

    return forecast_df


def optimize_trains(forecast_df: pd.DataFrame, station_flow: pd.DataFrame,
                    boarding_flow: pd.DataFrame) -> pd.DataFrame:
    """
    D02-D19列车配置优化
    基于预测客流量，为每趟列车确定:
    1. 最优编组数量（车厢数）
    2. 最优发车频次
    3. 停站方案
    """
    print("\n" + "=" * 60)
    print("步骤5: D02-D19列车配置优化")
    print("=" * 60)

    # 获取预测客流
    avg_daily_flow = forecast_df['预测客流'].mean()
    max_daily_flow = forecast_df['预测客流'].max()

    # 站点列表（按客流排序）
    top_stations = boarding_flow.head(15)['上车站'].tolist()

    # 列车角色分配
    # D02-D07: 高频干线列车（主要站点直达）
    # D08-D13: 中频区域列车（覆盖中等客流站点）
    # D14-D19: 低频支线列车（覆盖低客流站点）

    results = []
    for train in TRAIN_LIST:
        train_num = int(train[1:])

        # 确定列车角色
        if train_num <= 7:
            role = '干线高频'
            # 干线列车：高编组、高频次、主要站点
            carriages = 8 if max_daily_flow > 4000 else 6
            daily_freq = 12 if avg_daily_flow > 3000 else 10
            # 停靠主要站点（Top 5）
            stops = top_stations[:5]
            capacity_per_trip = carriages * 120  # 每车厢120人
        elif train_num <= 13:
            role = '区域中频'
            # 区域列车：中编组、中频次
            carriages = 6 if avg_daily_flow > 2500 else 4
            daily_freq = 8 if avg_daily_flow > 2500 else 6
            # 停靠中等站点
            stops = top_stations[3:10]
            capacity_per_trip = carriages * 120
        else:
            role = '支线低频'
            # 支线列车：低编组、低频次
            carriages = 4
            daily_freq = 4 if avg_daily_flow > 2000 else 3
            # 停靠所有站点
            stops = top_stations[8:]
            capacity_per_trip = carriages * 120

        # 计算日运输能力
        daily_capacity = capacity_per_trip * daily_freq

        # 计算客流分配（按角色权重）
        if role == '干线高频':
            flow_share = 0.45 / 6  # 干线占45%，6趟车
        elif role == '区域中频':
            flow_share = 0.35 / 6  # 区域占35%，6趟车
        else:
            flow_share = 0.20 / 6  # 支线占20%，6趟车

        assigned_flow = avg_daily_flow * flow_share
        load_factor = assigned_flow / daily_capacity if daily_capacity > 0 else 0

        # 根据负载率调整
        if load_factor > 0.85:
            # 超载，增加编组或频次
            if carriages < 8:
                carriages += 2
            else:
                daily_freq += 2
            daily_capacity = carriages * 120 * daily_freq
            load_factor = assigned_flow / daily_capacity

        # 峰时分布（早高峰7-9点，晚高峰17-19点）
        peak_hours = '7:00-9:00, 17:00-19:00'
        peak_freq = daily_freq // 2  # 高峰时段发车频次占一半

        results.append({
            '列车编号': train,
            '角色': role,
            '编组数量(车厢)': carriages,
            '每日发车频次': daily_freq,
            '高峰时段频次': peak_freq,
            '高峰时段': peak_hours,
            '停靠站点': ' → '.join(stops),
            '停靠站数': len(stops),
            '单次载客量': capacity_per_trip,
            '日运输能力': daily_capacity,
            '分配客流量': int(round(assigned_flow)),
            '负载率(%)': round(load_factor * 100, 1),
        })

    opt_df = pd.DataFrame(results)

    print("\n列车配置优化结果:")
    print(opt_df[['列车编号', '角色', '编组数量(车厢)', '每日发车频次',
                  '停靠站数', '日运输能力', '负载率(%)']].to_string(index=False))

    # 保存
    opt_path = os.path.join(OUTPUT_DIR, 'task3_optimization.csv')
    opt_df.to_csv(opt_path, index=False, encoding='utf-8-sig')
    print(f"\n优化结果已保存: {opt_path}")

    return opt_df


def plot_forecast_vs_actual(daily_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """图1: 预测 vs 实际对比图"""
    print("\n" + "=" * 60)
    print("步骤6: 生成可视化图表")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(14, 6))

    # 最近30天实际数据
    recent = daily_df.tail(30).copy()
    recent['日期'] = pd.to_datetime(recent['日期'])

    # 预测数据
    forecast = forecast_df.copy()
    forecast['日期'] = pd.to_datetime(forecast['日期'])

    ax.plot(recent['日期'], recent['客流人数'], 'b-o', label='历史实际客流',
            markersize=4, linewidth=1.5)
    ax.plot(forecast['日期'], forecast['预测客流'], 'r--s', label='预测客流',
            markersize=6, linewidth=2)

    # 添加分界线
    ax.axvline(x=recent['日期'].max(), color='gray', linestyle=':', alpha=0.7)
    ax.text(recent['日期'].max(), ax.get_ylim()[1] * 0.95, '  预测起点',
            fontsize=10, color='gray')

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('每日客流量', fontsize=12)
    ax.set_title('客流量预测 vs 历史实际对比', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(VIZ_DIR, 'forecast_vs_actual.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {path}")


def plot_forecast_trend(forecast_df: pd.DataFrame):
    """图2: 未来2周趋势图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    forecast = forecast_df.copy()
    forecast['日期'] = pd.to_datetime(forecast['日期'])
    forecast['星期'] = forecast['日期'].dt.dayofweek.map(
        {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
    )

    colors = ['#2196F3' if d < 5 else '#FF9800' for d in forecast['日期'].dt.dayofweek]
    bars = ax.bar(range(len(forecast)), forecast['预测客流'], color=colors, edgecolor='white')

    # 添加数值标签
    for bar, val in zip(bars, forecast['预测客流']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(val), ha='center', va='bottom', fontsize=9)

    # X轴标签
    labels = [f"{d.strftime('%m-%d')}\n{w}" for d, w in
              zip(forecast['日期'], forecast['星期'])]
    ax.set_xticks(range(len(forecast)))
    ax.set_xticklabels(labels, fontsize=9)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='工作日'),
                       Patch(facecolor='#FF9800', label='周末')]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('预测客流量', fontsize=12)
    ax.set_title('未来2周（2016-03-21 至 2016-04-03）客流趋势预测', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(VIZ_DIR, 'forecast_trend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {path}")


def plot_vehicle_config(opt_df: pd.DataFrame):
    """图3: 车辆配置图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 编组数量
    ax1 = axes[0, 0]
    colors = {'干线高频': '#E53935', '区域中频': '#1E88E5', '支线低频': '#43A047'}
    bar_colors = [colors[r] for r in opt_df['角色']]
    ax1.bar(opt_df['列车编号'], opt_df['编组数量(车厢)'], color=bar_colors, edgecolor='white')
    ax1.set_xlabel('列车编号', fontsize=10)
    ax1.set_ylabel('车厢数量', fontsize=10)
    ax1.set_title('各列车编组数量', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图2: 发车频次
    ax2 = axes[0, 1]
    ax2.bar(opt_df['列车编号'], opt_df['每日发车频次'], color=bar_colors, edgecolor='white')
    ax2.set_xlabel('列车编号', fontsize=10)
    ax2.set_ylabel('每日发车频次', fontsize=10)
    ax2.set_title('各列车每日发车频次', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # 子图3: 日运输能力
    ax3 = axes[1, 0]
    ax3.bar(opt_df['列车编号'], opt_df['日运输能力'], color=bar_colors, edgecolor='white')
    ax3.set_xlabel('列车编号', fontsize=10)
    ax3.set_ylabel('日运输能力（人次）', fontsize=10)
    ax3.set_title('各列车日运输能力', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # 子图4: 负载率
    ax4 = axes[1, 1]
    ax4.bar(opt_df['列车编号'], opt_df['负载率(%)'], color=bar_colors, edgecolor='white')
    ax4.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='超载阈值(85%)')
    ax4.set_xlabel('列车编号', fontsize=10)
    ax4.set_ylabel('负载率(%)', fontsize=10)
    ax4.set_title('各列车负载率', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10)

    plt.suptitle('D02-D19列车配置优化方案', fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(VIZ_DIR, 'vehicle_config.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {path}")


def plot_station_plan(opt_df: pd.DataFrame, boarding_flow: pd.DataFrame):
    """图4: 停站方案图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 各列车停靠站数
    ax1 = axes[0]
    colors = {'干线高频': '#E53935', '区域中频': '#1E88E5', '支线低频': '#43A047'}
    bar_colors = [colors[r] for r in opt_df['角色']]
    ax1.barh(opt_df['列车编号'], opt_df['停靠站数'], color=bar_colors, edgecolor='white')
    ax1.set_xlabel('停靠站数', fontsize=11)
    ax1.set_ylabel('列车编号', fontsize=11)
    ax1.set_title('各列车停靠站数', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='x')

    # 子图2: 站点客流热力图
    ax2 = axes[1]
    top10 = boarding_flow.head(10)
    ax2.barh(top10['上车站'], top10['平均日上车客流'], color='#42A5F5', edgecolor='white')
    ax2.set_xlabel('平均日上车客流量', fontsize=11)
    ax2.set_ylabel('站点编号', fontsize=11)
    ax2.set_title('Top 10 站点日均上车客流', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10)

    plt.suptitle('停站方案与站点客流分布', fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(VIZ_DIR, 'station_plan.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {path}")


def main():
    """主函数"""
    data_path = os.path.join(OUTPUT_DIR, 'task3_merged_data.csv')
    model_path = os.path.join(OUTPUT_DIR, 'task3_best_model.joblib')

    # 检查文件
    for p in [data_path, model_path]:
        if not os.path.exists(p):
            print(f"错误: 文件不存在: {p}")
            return

    # 加载模型
    print("加载最佳模型...")
    model = joblib.load(model_path)
    print(f"模型类型: {type(model).__name__}")

    # 1. 加载并聚合数据
    daily = load_and_aggregate(data_path)

    # 2. 特征工程
    df = engineer_features(daily)

    # 3. 加载站点信息
    station_flow, boarding_flow, alighting_flow = load_station_info(data_path)

    # 4. 预测未来2周
    forecast_df = forecast_future(model, df, n_days=14)

    # 5. D02-D19优化
    opt_df = optimize_trains(forecast_df, station_flow, boarding_flow)

    # 6. 可视化
    print("\n生成可视化图表...")
    plot_forecast_vs_actual(daily, forecast_df)
    plot_forecast_trend(forecast_df)
    plot_vehicle_config(opt_df)
    plot_station_plan(opt_df, boarding_flow)

    # 完成
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'task3_forecast.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'task3_optimization.csv')}")
    print(f"  - {VIZ_DIR}/forecast_vs_actual.png")
    print(f"  - {VIZ_DIR}/forecast_trend.png")
    print(f"  - {VIZ_DIR}/vehicle_config.png")
    print(f"  - {VIZ_DIR}/station_plan.png")


if __name__ == '__main__':
    main()
