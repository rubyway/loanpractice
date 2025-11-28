"""
Deep learning models for loan prediction using TensorFlow/Keras.
使用 TensorFlow/Keras 进行贷款预测的深度学习模型。
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def build_nn_model(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    """
    Build a neural network model for binary classification.
    构建用于二分类的神经网络模型。

    Args / 参数:
        input_dim: Number of input features / 输入特征数量
        hidden_layers: List of hidden layer sizes / 隐藏层大小列表
        dropout_rate: Dropout rate for regularization / 用于正则化的 Dropout 率

    Returns / 返回:
        Compiled Keras model / 编译后的 Keras 模型
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required for neural network models")

    model = Sequential()

    # Input layer / 输入层
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers / 隐藏层
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer / 输出层
    model.add(Dense(1, activation='sigmoid'))

    # Compile model / 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model


def train_nn_model(model, X_train, y_train, X_val=None, y_val=None,
                   epochs=100, batch_size=32, patience=10):
    """
    Train the neural network model.
    训练神经网络模型。

    Args / 参数:
        model: Keras model / Keras 模型
        X_train: Training features / 训练特征
        y_train: Training target / 训练目标
        X_val: Validation features (optional) / 验证特征（可选）
        y_val: Validation target (optional) / 验证目标（可选）
        epochs: Maximum number of epochs / 最大训练轮数
        batch_size: Batch size for training / 训练批次大小
        patience: Early stopping patience / 早停耐心值

    Returns / 返回:
        Trained model and training history / 已训练的模型和训练历史
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6
        )
    ]

    validation_data = (X_val, y_val) if X_val is not None else None

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_nn_model(model, X, y):
    """
    Evaluate neural network model.
    评估神经网络模型。

    Args / 参数:
        model: Trained Keras model / 已训练的 Keras 模型
        X: Features / 特征
        y: Target / 目标

    Returns / 返回:
        Dictionary of metrics / 指标字典
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Get predictions / 获取预测
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y, y_pred, average='binary', zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob)
    }

    return metrics


def build_deep_nn_model(input_dim):
    """
    Build a deeper neural network for more complex patterns.
    构建更深的神经网络以处理更复杂的模式。

    Args / 参数:
        input_dim: Number of input features / 输入特征数量

    Returns / 返回:
        Compiled Keras model / 编译后的 Keras 模型
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required for neural network models")

    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model
