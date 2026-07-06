"""
任务3: 客流量预测模型
使用随机森林、XGBoost和线性回归预测每日客流量
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
EDA_DIR = os.path.join(OUTPUT_DIR, 'task3_eda')
os.makedirs(EDA_DIR, exist_ok=True)


def load_and_aggregate(path: str) -> pd.DataFrame:
    """加载数据并按日聚合客流量"""
    print("=" * 60)
    print("步骤1: 加载并聚合数据")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"原始数据形状: {df.shape}")
    print(f"列: {df.columns.tolist()}")

    # 解析日期
    df['日期'] = pd.to_datetime(df['日期'])

    # 提取每日天气信息（取每行第一条非空天气）
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
    print(f"客流量统计:\n{daily['客流人数'].describe()}")
    return daily


def engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """特征工程：时间特征、滞后特征、天气特征"""
    print("\n" + "=" * 60)
    print("步骤2: 特征工程")
    print("=" * 60)

    df = daily.copy()

    # --- 时间特征 ---
    df['day_of_week'] = df['日期'].dt.dayofweek      # 0=周一, 6=周日
    df['month'] = df['日期'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_year'] = df['日期'].dt.dayofyear
    df['day_of_month'] = df['日期'].dt.day

    # --- 滞后特征：7日滚动平均 ---
    df['rolling_7d_mean'] = df['客流人数'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_7d_std'] = df['客流人数'].shift(1).rolling(window=7, min_periods=1).std()
    df['lag_1'] = df['客流人数'].shift(1)
    df['lag_7'] = df['客流人数'].shift(7)

    # --- 天气编码 ---
    # 将天气文本编码为数值（白天和夜间联合fit，避免unseen labels）
    le_weather = LabelEncoder()
    if df['天气_白天'].notna().any():
        all_weather = pd.concat([df['天气_白天'].fillna('未知'), df['天气_夜间'].fillna('未知')]).unique()
        le_weather.fit(all_weather)
        df['天气_白天_enc'] = le_weather.transform(df['天气_白天'].fillna('未知'))
        df['天气_夜间_enc'] = le_weather.transform(df['天气_夜间'].fillna('未知'))
    else:
        df['天气_白天_enc'] = 0
        df['天气_夜间_enc'] = 0

    # 风力编码（白天和夜间联合fit）
    le_wind = LabelEncoder()
    if df['风力_白天'].notna().any():
        all_wind = pd.concat([df['风力_白天'].fillna('未知'), df['风力_夜间'].fillna('未知')]).unique()
        le_wind.fit(all_wind)
        df['风力_白天_enc'] = le_wind.transform(df['风力_白天'].fillna('未知'))
        df['风力_夜间_enc'] = le_wind.transform(df['风力_夜间'].fillna('未知'))
    else:
        df['风力_白天_enc'] = 0
        df['风力_夜间_enc'] = 0

    # 温度填充缺失值
    df['最高温'] = df['最高温'].interpolate(method='linear').fillna(df['最高温'].median())
    df['最低温'] = df['最低温'].interpolate(method='linear').fillna(df['最低温'].median())

    # 删除因滞后特征产生的NaN行（前7天）
    df = df.dropna(subset=['rolling_7d_mean', 'lag_7']).reset_index(drop=True)

    print(f"特征工程后数据形状: {df.shape}")
    print(f"特征列: {[c for c in df.columns if c not in ['日期', '客流人数']]}")
    return df


def prepare_train_test(df: pd.DataFrame):
    """准备训练集和测试集"""
    print("\n" + "=" * 60)
    print("步骤3: 划分训练集/测试集")
    print("=" * 60)

    # 特征列
    feature_cols = [
        'day_of_week', 'month', 'is_weekend', 'day_of_year', 'day_of_month',
        'rolling_7d_mean', 'rolling_7d_std', 'lag_1', 'lag_7',
        '最高温', '最低温',
        '天气_白天_enc', '天气_夜间_enc',
        '风力_白天_enc', '风力_夜间_enc',
        '车次数', '上车站数', '下车站数',
    ]

    # 训练集: 2015年数据, 测试集: 2016年1-3月
    train_mask = df['日期'].dt.year == 2015
    test_mask = df['日期'].dt.year == 2016

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, '客流人数'].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, '客流人数'].values

    dates_train = df.loc[train_mask, '日期'].values
    dates_test = df.loc[test_mask, '日期'].values

    print(f"训练集: {X_train.shape[0]} 天 (2015年)")
    print(f"测试集: {X_test.shape[0]} 天 (2016年1-3月)")
    print(f"特征数: {X_train.shape[1]}")
    return X_train, y_train, X_test, y_test, dates_train, dates_test, feature_cols


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """计算评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE: 避免除以0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"\n{name}:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAPE = {mape:.2f}%")

    return {'模型': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE(%)': mape}


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练三种模型并评估"""
    print("\n" + "=" * 60)
    print("步骤4: 训练模型并评估")
    print("=" * 60)

    results = []
    predictions = {}

    # 1. 线性回归（基线）
    print("\n--- 训练线性回归 ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_lr, '线性回归'))
    predictions['线性回归'] = y_pred_lr

    # 2. 随机森林
    print("\n--- 训练随机森林 ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_rf, '随机森林'))
    predictions['随机森林'] = y_pred_rf

    # 3. XGBoost
    print("\n--- 训练XGBoost ---")
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_xgb, 'XGBoost'))
    predictions['XGBoost'] = y_pred_xgb

    # 模型字典，用于保存最佳模型
    models = {'线性回归': lr, '随机森林': rf, 'XGBoost': xgb}

    return results, predictions, models


def save_results(results: list, predictions: dict, y_test: np.ndarray,
                 dates_test: np.ndarray, models: dict):
    """保存结果：CSV、图表、模型"""
    print("\n" + "=" * 60)
    print("步骤5: 保存结果")
    print("=" * 60)

    # --- 模型对比表 ---
    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(OUTPUT_DIR, 'task3_model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"模型对比表已保存: {comparison_path}")
    print(f"\n{comparison_df.to_string(index=False)}")

    # --- 最佳模型指标 ---
    best_idx = comparison_df['R²'].idxmax()
    best_name = comparison_df.loc[best_idx, '模型']
    best_metrics = comparison_df.loc[[best_idx]].copy()
    best_metrics_path = os.path.join(OUTPUT_DIR, 'task3_model_metrics.csv')
    best_metrics.to_csv(best_metrics_path, index=False, encoding='utf-8-sig')
    print(f"\n最佳模型: {best_name}")
    print(f"最佳模型指标已保存: {best_metrics_path}")

    # --- 保存最佳模型 ---
    best_model = models[best_name]
    model_path = os.path.join(OUTPUT_DIR, 'task3_best_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"最佳模型已保存: {model_path}")

    # --- 预测 vs 实际 图 ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 子图1: 时间序列对比
    ax1 = axes[0]
    dates_plot = pd.to_datetime(dates_test)
    ax1.plot(dates_plot, y_test, 'b-o', label='实际客流', markersize=3, linewidth=1.2)
    colors = {'线性回归': 'red', '随机森林': 'green', 'XGBoost': 'orange'}
    for name, pred in predictions.items():
        ax1.plot(dates_plot, pred, '--', label=f'{name}预测', color=colors[name], linewidth=1)
    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_ylabel('每日客流人数', fontsize=11)
    ax1.set_title('2016年1-3月客流量预测 vs 实际', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2: 散点图（最佳模型）
    ax2 = axes[1]
    best_pred = predictions[best_name]
    ax2.scatter(y_test, best_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=40)
    min_val = min(y_test.min(), best_pred.min())
    max_val = max(y_test.max(), best_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='完美预测线')
    ax2.set_xlabel('实际客流人数', fontsize=11)
    ax2.set_ylabel('预测客流人数', fontsize=11)
    ax2.set_title(f'{best_name}模型: 预测 vs 实际散点图 (R²={results[best_idx]["R²"]:.4f})', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(EDA_DIR, '11_prediction_vs_actual.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测对比图已保存: {plot_path}")

    # --- 额外保存特征重要性图（树模型） ---
    if best_name in ['随机森林', 'XGBoost']:
        fig2, ax3 = plt.subplots(figsize=(10, 6))
        feature_cols = [
            'day_of_week', 'month', 'is_weekend', 'day_of_year', 'day_of_month',
            'rolling_7d_mean', 'rolling_7d_std', 'lag_1', 'lag_7',
            '最高温', '最低温',
            '天气_白天_enc', '天气_夜间_enc',
            '风力_白天_enc', '风力_夜间_enc',
            '车次数', '上车站数', '下车站数',
        ]
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)
        ax3.barh(range(len(sorted_idx)), importances[sorted_idx])
        ax3.set_yticks(range(len(sorted_idx)))
        ax3.set_yticklabels([feature_cols[i] for i in sorted_idx])
        ax3.set_xlabel('特征重要性', fontsize=11)
        ax3.set_title(f'{best_name}模型特征重要性', fontsize=13)
        plt.tight_layout()
        fi_path = os.path.join(EDA_DIR, '12_feature_importance.png')
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存: {fi_path}")

    return best_name


def main():
    """主函数"""
    data_path = os.path.join(os.path.dirname(__file__), 'output', 'task3_merged_data.csv')
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return

    # 1. 加载并聚合
    daily = load_and_aggregate(data_path)

    # 2. 特征工程
    df = engineer_features(daily)

    # 3. 准备数据
    X_train, y_train, X_test, y_test, dates_train, dates_test, feature_cols = prepare_train_test(df)

    # 4. 训练并评估
    results, predictions, models = train_and_evaluate(X_train, y_train, X_test, y_test)

    # 5. 保存结果
    best_name = save_results(results, predictions, y_test, dates_test, models)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"最佳模型: {best_name}")
    print(f"输出文件:")
    print(f"  - task3_model_comparison.csv  (模型对比)")
    print(f"  - task3_model_metrics.csv     (最佳模型指标)")
    print(f"  - task3_best_model.joblib     (保存的模型)")
    print(f"  - task3_eda/11_prediction_vs_actual.png (预测对比图)")
    print(f"  - task3_eda/12_feature_importance.png   (特征重要性图)")


if __name__ == '__main__':
    main()
