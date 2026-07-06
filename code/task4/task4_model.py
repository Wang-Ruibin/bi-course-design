#!/usr/bin/env python3
"""
任务4：故障报警分类模型
- 二分类：预测是否发生故障（故障标记）
- 多标签分类：预测具体故障类型
- 模型：随机森林、XGBoost、逻辑回归
"""

import os
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 中文字体设置 ──────────────────────────────────────────────────
def setup_chinese_font():
    """设置 matplotlib 中文字体"""
    font_candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            prop = font_manager.FontProperties(fname=fp)
            matplotlib.rcParams["font.family"] = prop.get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用: {fp}")
            return
    # fallback: try system sans-serif
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("[字体] 未找到中文字体，使用默认 sans-serif")

setup_chinese_font()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────
ROOT = Path("/mnt/d/resource/课程/商务智能应用/课设")
DATA_PATH = ROOT / "code" / "output" / "task4_cleaned_data.csv"
OUTPUT_DIR = ROOT / "code" / "output"
EDA_DIR = OUTPUT_DIR / "task4_eda"
MODEL_DIR = OUTPUT_DIR / "task4_models"
EDA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. 加载数据 ────────────────────────────────────────────────────
print("=" * 60)
print("步骤 1：加载数据")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"数据形状: {df.shape}")
print(f"故障标记分布:\n{df['故障标记'].value_counts()}")
print(f"故障率: {df['故障标记'].mean():.4%}")

# ── 2. 特征工程 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 2：特征工程")
print("=" * 60)

# 传感器特征列（排除日期、时间、生产线编号、故障列、目标列）
SENSOR_COLS = [
    "物料推送气缸推送状态", "物料推送气缸收回状态", "物料推送数", "物料待抓取数",
    "放置容器数", "容器上传检测数", "填装检测数", "填装定位器固定状态", "填装定位器放开状态",
    "物料抓取数", "填装旋转数", "填装下降数", "填装数", "加盖检测数", "加盖定位数",
    "推盖数", "加盖下降数", "加盖数", "拧盖检测数", "拧盖定位数", "拧盖下降数",
    "拧盖旋转数", "拧盖数", "合格数", "不合格数",
]

# 故障类型列（用于多标签分类）
FAULT_COLS = [
    "物料推送装置故障1001", "物料检测装置故障2001", "填装装置检测故障4001",
    "填装装置定位故障4002", "填装装置填装故障4003", "加盖装置定位故障5001",
    "加盖装置加盖故障5002", "拧盖装置定位故障6001", "拧盖装置拧盖故障6002",
]

# 额外特征
EXTRA_FEATURES = ["hour_of_day", "is_daytime"]

ALL_FEATURES = SENSOR_COLS + EXTRA_FEATURES

# 将生产线编号转为 one-hot
df_encoded = pd.get_dummies(df["生产线编号"], prefix="line")
X_base = pd.concat([df[ALL_FEATURES], df_encoded], axis=1)

# 二分类目标
y_binary = df["故障标记"]

# 多标签目标：将故障计数转为二值（>0 表示有该类故障）
y_multi = (df[FAULT_COLS] > 0).astype(int)

print(f"特征数量: {X_base.shape[1]}")
print(f"特征列: {list(X_base.columns)}")
print(f"多标签故障分布:")
for col in FAULT_COLS:
    count = y_multi[col].sum()
    print(f"  {col}: {count} ({count / len(df):.2%})")

# ── 3. 数据划分与标准化 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 3：数据划分与标准化")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"训练集: {X_train.shape[0]} 样本, 故障率 {y_train.mean():.4%}")
print(f"测试集: {X_test.shape[0]} 样本, 故障率 {y_test.mean():.4%}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 类别权重
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"正负样本比: 1:{scale_pos_weight:.1f}")

# ── 4. 训练模型（二分类）────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 4：训练二分类模型")
print("=" * 60)

results = {}

# --- 4.1 逻辑回归（基线）---
print("\n[模型 1] 逻辑回归 (baseline)...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"  AUC-ROC: {auc_lr:.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["正常", "故障"]))

results["逻辑回归"] = {
    "model": lr,
    "y_pred": y_pred_lr,
    "y_prob": y_prob_lr,
    "auc": auc_lr,
}

# --- 4.2 随机森林 ---
print("\n[模型 2] 随机森林...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    max_depth=15,
    min_samples_leaf=5,
)
rf.fit(X_train, y_train)  # 树模型不需要标准化
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"  AUC-ROC: {auc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=["正常", "故障"]))

results["随机森林"] = {
    "model": rf,
    "y_pred": y_pred_rf,
    "y_prob": y_prob_rf,
    "auc": auc_rf,
}

# --- 4.3 XGBoost ---
print("\n[模型 3] XGBoost...")
try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train, verbose=False)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    print(f"  AUC-ROC: {auc_xgb:.4f}")
    print(classification_report(y_test, y_pred_xgb, target_names=["正常", "故障"]))

    results["XGBoost"] = {
        "model": xgb,
        "y_pred": y_pred_xgb,
        "y_prob": y_prob_xgb,
        "auc": auc_xgb,
    }
except ImportError:
    print("  [跳过] xgboost 未安装，跳过 XGBoost 模型")

# ── 5. 模型评估汇总 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 5：模型评估汇总")
print("=" * 60)

metrics_rows = []
for name, res in results.items():
    from sklearn.metrics import precision_score, recall_score, f1_score

    prec = precision_score(y_test, res["y_pred"])
    rec = recall_score(y_test, res["y_pred"])
    f1 = f1_score(y_test, res["y_pred"])
    auc = res["auc"]
    metrics_rows.append({
        "模型": name,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "AUC-ROC": round(auc, 4),
    })
    print(f"{name}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUTPUT_DIR / "task4_model_metrics.csv", index=False, encoding="utf-8-sig")
print(f"\n评估指标已保存: {OUTPUT_DIR / 'task4_model_metrics.csv'}")

# ── 6. ROC 曲线 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 6：绘制 ROC 曲线")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#2196F3", "#4CAF50", "#FF9800"]
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={res["auc"]:.4f})')

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("假阳率 (FPR)")
ax.set_ylabel("真阳率 (TPR)")
ax.set_title("故障报警分类 — ROC 曲线对比")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(EDA_DIR / "09_roc_curve.png", dpi=150)
plt.close(fig)
print(f"ROC 曲线已保存: {EDA_DIR / '09_roc_curve.png'}")

# ── 7. 特征重要性 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 7：特征重要性分析")
print("=" * 60)

# 使用随机森林的特征重要性
if "随机森林" in results:
    importances = results["随机森林"]["model"].feature_importances_
    feat_names = X_base.columns
    feat_imp_df = pd.DataFrame({
        "特征": feat_names,
        "重要性": importances,
    }).sort_values("重要性", ascending=False)

    top_n = 20
    top_feats = feat_imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), top_feats["重要性"].values[::-1], color="#4CAF50", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats["特征"].values[::-1])
    ax.set_xlabel("特征重要性 (Gini)")
    ax.set_title(f"随机森林 — Top {top_n} 特征重要性")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "10_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"特征重要性图已保存: {EDA_DIR / '10_feature_importance.png'}")

    print(f"\nTop 10 重要特征:")
    for _, row in feat_imp_df.head(10).iterrows():
        print(f"  {row['特征']}: {row['重要性']:.4f}")

# ── 8. 多标签分类（可选）────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 8：多标签故障分类")
print("=" * 60)

try:
    # 只对故障样本做多标签（有故障标记=1 的行）
    fault_mask = df["故障标记"] == 1
    X_multi = X_base[fault_mask]
    y_multi_fault = y_multi[fault_mask]
    print(f"故障样本数: {X_multi.shape[0]}")
    print(f"多标签分布 (故障样本内):")
    for col in FAULT_COLS:
        print(f"  {col}: {y_multi_fault[col].sum()} ({y_multi_fault[col].mean():.2%})")

    # 训练多标签随机森林
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_multi, y_multi_fault, test_size=0.2, random_state=42
    )

    rf_multi = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        n_jobs=-1,
    )
    rf_multi.fit(X_m_train, y_m_train)
    y_m_pred = rf_multi.predict(X_m_test)

    from sklearn.metrics import f1_score as f1s

    print("\n多标签分类结果 (故障样本内的各故障类型):")
    for i, col in enumerate(FAULT_COLS):
        f1 = f1s(y_m_test.iloc[:, i], y_m_pred[:, i], zero_division=0)
        prec = precision_score(y_m_test.iloc[:, i], y_m_pred[:, i], zero_division=0)
        rec = recall_score(y_m_test.iloc[:, i], y_m_pred[:, i], zero_division=0)
        print(f"  {col}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

except Exception as e:
    print(f"  多标签分类失败: {e}")

# ── 9. 保存最佳模型 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 9：保存最佳模型")
print("=" * 60)

# 选择 AUC 最高的模型
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
best_auc = results[best_name]["auc"]

model_path = MODEL_DIR / "best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model": best_model,
        "model_name": best_name,
        "auc": best_auc,
        "feature_names": list(X_base.columns),
        "scaler": scaler if best_name == "逻辑回归" else None,
    }, f)

print(f"最佳模型: {best_name} (AUC={best_auc:.4f})")
print(f"模型已保存: {model_path}")

# ── 10. 总结 ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("完成！输出文件:")
print("=" * 60)
print(f"  1. 模型指标: {OUTPUT_DIR / 'task4_model_metrics.csv'}")
print(f"  2. ROC 曲线: {EDA_DIR / '09_roc_curve.png'}")
print(f"  3. 特征重要性: {EDA_DIR / '10_feature_importance.png'}")
print(f"  4. 最佳模型: {model_path}")
