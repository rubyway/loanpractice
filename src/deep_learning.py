"""
Deep learning models for loan prediction using TensorFlow/Keras.
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

    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled Keras model
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required for neural network models")

    model = Sequential()

    # Input layer
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
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

    Args:
        model: Keras model
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        patience: Early stopping patience

    Returns:
        Trained model and training history
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

    Args:
        model: Trained Keras model
        X: Features
        y: Target

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Get predictions
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

    Args:
        input_dim: Number of input features

    Returns:
        Compiled Keras model
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
