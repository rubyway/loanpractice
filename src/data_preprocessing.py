"""
Data loading and preprocessing module for loan prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(train_path, test_path, original_path=None):
    """
    Load train, test and optionally original loan data.

    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        original_path: Optional path to original loan data CSV file

    Returns:
        Tuple of (train_df, test_df, original_df or None)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    original_df = None
    if original_path:
        original_df = pd.read_csv(original_path)

    return train_df, test_df, original_df


def preprocess_data(train_df, test_df, target_col='loan_status'):
    """
    Preprocess the data for machine learning.

    This function:
    - Handles missing values
    - Encodes categorical variables
    - Scales numerical features

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (X_train, y_train, X_test, encoders, scaler, test_feature_cols, test_index)
    """
    # Make copies to avoid modifying original data
    train = train_df.copy()
    test = test_df.copy()

    # Store test index for later submission
    test_index = test.index if 'id' not in test.columns else test['id']

    # Identify categorical and numerical columns
    # Exclude target column and id column
    exclude_cols = [target_col, 'id', 'Id', 'ID']

    # Get feature columns
    feature_cols = [col for col in train.columns if col not in exclude_cols]
    categorical_cols = train[feature_cols].select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train[feature_cols].select_dtypes(include=['number']).columns.tolist()

    # Handle missing values
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

    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to handle unseen categories
        if col in test.columns:
            combined = pd.concat([train[col].astype(str), test[col].astype(str)])
        else:
            combined = train[col].astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        if col in test.columns:
            test[col] = le.transform(test[col].astype(str))
        encoders[col] = le

    # Prepare features and target
    X_train = train[feature_cols]
    y_train = train[target_col] if target_col in train.columns else None

    # Handle test features - only use columns that exist in test
    test_feature_cols = [col for col in feature_cols if col in test.columns]
    X_test = test[test_feature_cols]

    # Ensure X_train only has columns that also exist in X_test
    X_train = X_train[test_feature_cols]

    # Scale numerical features
    scaler = StandardScaler()
    num_cols_in_features = [col for col in numerical_cols if col in test_feature_cols]
    if num_cols_in_features:
        X_train.loc[:, num_cols_in_features] = scaler.fit_transform(X_train[num_cols_in_features])
        X_test.loc[:, num_cols_in_features] = scaler.transform(X_test[num_cols_in_features])

    return X_train, y_train, X_test, encoders, scaler, test_feature_cols, test_index


def get_train_val_split(X, y, val_size=0.2, random_state=42):
    """
    Split training data into train and validation sets.

    Args:
        X: Feature DataFrame
        y: Target Series
        val_size: Validation set size ratio
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    return train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)
