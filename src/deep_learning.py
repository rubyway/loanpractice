"""
Deep learning models for loan prediction using PyTorch.
使用 PyTorch 实现贷款预测的深度学习模型。
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:  # pragma: no cover - handled gracefully for environments without torch
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Adam = None  # type: ignore
    ReduceLROnPlateau = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore


if HAS_TORCH:
    class FeedForwardNN(nn.Module):
        """Simple feed-forward neural network with batch norm and dropout."""

        def __init__(self, input_dim: int, hidden_layers: Sequence[int], dropout_rate: float) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            in_features = input_dim
            for units in hidden_layers:
                layers.append(nn.Linear(in_features, units))
                layers.append(nn.BatchNorm1d(units))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                in_features = units
            layers.append(nn.Linear(in_features, 1))
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.model(x)


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for neural network models")


def _to_numpy(array_like: Iterable, dtype=np.float32) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        arr = array_like
    else:
        arr = np.asarray(array_like)
    return arr.astype(dtype, copy=False)


def _prepare_tensor(data, *, is_target: bool = False) -> "torch.Tensor":
    _require_torch()
    arr = _to_numpy(data, dtype=np.float32)
    if is_target:
        arr = arr.reshape(-1, 1)
    return torch.from_numpy(arr)


def build_nn_model(input_dim: int, hidden_layers: Sequence[int] = (128, 64, 32),
                   dropout_rate: float = 0.3) -> "FeedForwardNN":
    """
    Build a PyTorch neural network for binary classification.
    构建用于二分类的 PyTorch 神经网络。
    """
    _require_torch()
    return FeedForwardNN(input_dim, hidden_layers, dropout_rate)


def build_deep_nn_model(input_dim: int) -> "FeedForwardNN":
    """
    Build a deeper PyTorch neural network for more complex patterns.
    构建更深的 PyTorch 神经网络。
    """
    return build_nn_model(input_dim, hidden_layers=(256, 128, 64, 32, 16), dropout_rate=0.3)


def train_nn_model(model: "FeedForwardNN", X_train, y_train, X_val=None, y_val=None,
                   epochs: int = 100, batch_size: int = 32, patience: int = 10,
                   learning_rate: float = 1e-3, device: Optional[str] = None) -> Tuple["FeedForwardNN", Dict[str, List[float]]]:
    """
    Train the PyTorch neural network model with early stopping and LR scheduling.
    使用早停与学习率调度训练 PyTorch 神经网络模型。
    """
    _require_torch()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_train_tensor = _prepare_tensor(X_train)
    y_train_tensor = _prepare_tensor(y_train, is_target=True)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_tensor = _prepare_tensor(X_val)
        y_val_tensor = _prepare_tensor(y_val, is_target=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if val_loader is not None:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=max(1, patience // 2), min_lr=1e-6)

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_running_loss += loss.item() * batch_X.size(0)
            val_loss = val_running_loss / len(val_loader.dataset)
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Simple early stopping on validation loss / 基于验证损失的早停
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to("cpu")
    return model, history


def predict_nn_model(model: "FeedForwardNN", X, batch_size: int = 256,
                     device: Optional[str] = None) -> np.ndarray:
    """Generate probability predictions from the PyTorch model."""
    _require_torch()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_was_training = model.training
    model.eval()

    X_tensor = _prepare_tensor(X)
    data_loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)

    probs: List[np.ndarray] = []
    with torch.no_grad():
        for (batch_X,) in data_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(batch_probs)

    if model_was_training:
        model.train()
    model = model.to("cpu")

    return np.concatenate(probs)


def evaluate_nn_model(model: "FeedForwardNN", X, y) -> Dict[str, float]:
    """
    Evaluate neural network model using standard classification metrics.
    使用分类指标评估神经网络模型。
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_prob = predict_nn_model(model, X)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average='binary', zero_division=0),
        "recall": recall_score(y, y_pred, average='binary', zero_division=0),
        "f1": f1_score(y, y_pred, average='binary', zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    return metrics
