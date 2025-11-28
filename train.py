#!/usr/bin/env python
"""
Quick training script for loan prediction.
This script provides a simplified interface for training and prediction.

贷款预测快速训练脚本。
本脚本提供了简化的训练和预测接口。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Try to import advanced libraries / 尝试导入高级库
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available")


def quick_preprocess(train_df, test_df, target_col='loan_status'):
    """
    Quick preprocessing of loan data.
    贷款数据快速预处理。
    """
    train = train_df.copy()
    test = test_df.copy()

    # Get feature columns / 获取特征列
    exclude_cols = [target_col, 'id', 'Id', 'ID']
    feature_cols = [col for col in train.columns if col not in exclude_cols]

    # Identify column types / 识别列类型
    cat_cols = train[feature_cols].select_dtypes(include=['object']).columns.tolist()
    num_cols = train[feature_cols].select_dtypes(include=['number']).columns.tolist()

    # Handle missing values / 处理缺失值
    for col in num_cols:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        if col in test.columns:
            test[col] = test[col].fillna(median)

    for col in cat_cols:
        mode = train[col].mode()[0] if len(train[col].mode()) > 0 else 'Unknown'
        train[col] = train[col].fillna(mode)
        if col in test.columns:
            test[col] = test[col].fillna(mode)

    # Encode categorical variables / 编码分类变量
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        if col in test.columns:
            combined = pd.concat([train[col].astype(str), test[col].astype(str)])
        else:
            combined = train[col].astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))
        encoders[col] = le

    # Get common features / 获取公共特征
    test_feature_cols = [col for col in feature_cols if col in test.columns]

    X_train = train[test_feature_cols]
    y_train = train[target_col]
    X_test = test[test_feature_cols]

    # Scale features / 缩放特征
    scaler = StandardScaler()
    num_features = [col for col in num_cols if col in test_feature_cols]
    if num_features:
        X_train.loc[:, num_features] = scaler.fit_transform(X_train[num_features])
        X_test.loc[:, num_features] = scaler.transform(X_test[num_features])

    # Get test ids / 获取测试集ID
    if 'id' in test.columns:
        test_ids = test['id']
    elif 'Id' in test.columns:
        test_ids = test['Id']
    else:
        test_ids = pd.Series(range(len(test)))

    return X_train, y_train, X_test, test_ids


def train_and_predict(train_path, test_path, output_path='predictions.csv'):
    """
    Train models and generate predictions.
    训练模型并生成预测。

    Args / 参数:
        train_path: Path to training CSV / 训练 CSV 文件路径
        test_path: Path to test CSV / 测试 CSV 文件路径
        output_path: Path for output predictions / 预测输出文件路径
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("\nPreprocessing data...")
    X_train, y_train, X_test, test_ids = quick_preprocess(train_df, test_df)

    # Split for validation / 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training set: {len(X_tr)}, Validation set: {len(X_val)}")

    # Define models / 定义模型
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=7,
            random_state=42, eval_metric='logloss'
        )

    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=7,
            random_state=42, verbose=-1
        )

    # Train and evaluate / 训练和评估
    print("\nTraining models...")
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        results[name] = {'model': model, 'accuracy': acc, 'f1': f1, 'roc_auc': auc}
        print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    # Find best model / 找到最佳模型
    best_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_name]['model']
    print(f"\nBest model: {best_name}")

    # Retrain best model on full data / 在完整数据上重新训练最佳模型
    print("\nRetraining on full training data...")
    best_model.fit(X_train, y_train)

    # Generate predictions / 生成预测
    print("Generating predictions...")
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    # Save predictions / 保存预测
    output = pd.DataFrame({
        'id': test_ids,
        'loan_status': predictions
    })
    output.to_csv(output_path, index=False)

    print(f"\nPredictions saved to {output_path}")
    print(f"Predicted 0: {(predictions == 0).sum()}")
    print(f"Predicted 1: {(predictions == 1).sum()}")

    return output


if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'predictions.csv'
        train_and_predict(train_path, test_path, output_path)
    else:
        print("Usage: python train.py <train.csv> <test.csv> [output.csv]")
        print("\nExample:")
        print("  python train.py data/train.csv data/test.csv predictions.csv")
