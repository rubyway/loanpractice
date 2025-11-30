"""
Data loading and preprocessing module for loan prediction.
贷款预测的数据加载和预处理模块。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(train_path, test_path, original_path=None):
    """
    Load train, test and optionally original loan data.
    加载训练、测试和可选的原始贷款数据。

    Args / 参数:
        train_path: Path to training CSV file / 训练 CSV 文件路径
        test_path: Path to test CSV file / 测试 CSV 文件路径
        original_path: Optional path to original loan data CSV file / 可选的原始贷款数据 CSV 文件路径

    Returns / 返回:
        Tuple of (train_df, test_df, original_df or None) / (train_df, test_df, original_df 或 None) 元组
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    original_df = None
    if original_path:
        original_df = pd.read_csv(original_path)

    return train_df, test_df, original_df


def preprocess_data(train_df, test_df, target_col='loan_paid_back'):
    """
    Preprocess the data for machine learning.
    为机器学习预处理数据。

    This function: / 本函数：
    - Handles missing values / 处理缺失值
    - Encodes categorical variables / 编码分类变量
    - Scales numerical features / 缩放数值特征

    Args / 参数:
        train_df: Training DataFrame / 训练 DataFrame
        test_df: Test DataFrame / 测试 DataFrame
        target_col: Name of target column / 目标列名称

    Returns / 返回:
        Tuple of (X_train, y_train, X_test, encoders, scaler, test_feature_cols, test_index)
        (X_train, y_train, X_test, 编码器, 缩放器, 测试特征列, 测试索引) 元组
    """
    # Make copies to avoid modifying original data / 创建副本以避免修改原始数据
    train = train_df.copy()
    test = test_df.copy()

    # Store test index for later submission / 保存测试索引以供后续提交使用
    test_index = test.index if 'id' not in test.columns else test['id']

    # Identify categorical and numerical columns / 识别分类列和数值列
    # Exclude target column and id column / 排除目标列和 id 列
    exclude_cols = [target_col, 'id', 'Id', 'ID']

    # Get feature columns / 获取特征列
    feature_cols = [col for col in train.columns if col not in exclude_cols]
    categorical_cols = train[feature_cols].select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train[feature_cols].select_dtypes(include=['number']).columns.tolist()

    # Handle missing values / 处理缺失值
    for col in numerical_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)

    for col in categorical_cols:
        mode_val = train[col].mode()[0] if len(train[col].mode()) > 0 else 'Unknown'
        train[col] = train[col].fillna(mode_val)
        if col in test.columns:
            test[col] = test[col].fillna(mode_val)

    # Encode categorical variables / 编码分类变量
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to handle unseen categories / 在合并数据上拟合以处理未见过的类别
        if col in test.columns:
            combined = pd.concat([train[col].astype(str), test[col].astype(str)])
        else:
            combined = train[col].astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))
        encoders[col] = le

    # Prepare features and target / 准备特征和目标
    X_train = train[feature_cols]
    y_train = train[target_col].astype(int) if target_col in train.columns else None

    # Handle test features - only use columns that exist in test / 处理测试特征 - 只使用测试集中存在的列
    test_feature_cols = [col for col in feature_cols if col in test.columns]
    X_test = test[test_feature_cols]

    # Ensure X_train only has columns that also exist in X_test / 确保 X_train 只包含 X_test 中也存在的列
    X_train = X_train[test_feature_cols]

    # Scale numerical features / 缩放数值特征
    scaler = StandardScaler()
    num_cols_in_features = [col for col in numerical_cols if col in test_feature_cols]
    if num_cols_in_features:
        X_train = X_train.copy()
        X_test = X_test.copy()
        # Ensure float dtype before scaling to avoid pandas FutureWarning / 缩放前先转为浮点以避免 FutureWarning
        X_train.loc[:, num_cols_in_features] = X_train[num_cols_in_features].astype(np.float64)
        X_test.loc[:, num_cols_in_features] = X_test[num_cols_in_features].astype(np.float64)
        X_train.loc[:, num_cols_in_features] = scaler.fit_transform(X_train[num_cols_in_features])
        X_test.loc[:, num_cols_in_features] = scaler.transform(X_test[num_cols_in_features])

    return X_train, y_train, X_test, encoders, scaler, test_feature_cols, test_index


def get_train_val_split(X, y, val_size=0.2, random_state=42):
    """
    Split training data into train and validation sets.
    将训练数据划分为训练集和验证集。

    Args / 参数:
        X: Feature DataFrame / 特征 DataFrame
        y: Target Series / 目标 Series
        val_size: Validation set size ratio / 验证集大小比例
        random_state: Random seed / 随机种子

    Returns / 返回:
        Tuple of (X_train, X_val, y_train, y_val) / (X_train, X_val, y_train, y_val) 元组
    """
    return train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)
