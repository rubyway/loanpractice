"""
Machine Learning models for loan prediction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_models():
    """
    Get dictionary of all available models with default parameters.

    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'svm': SVC(probability=True, random_state=42),
    }

    if HAS_XGBOOST:
        models['xgboost'] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss'
        )

    if HAS_LIGHTGBM:
        models['lightgbm'] = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )

    if HAS_CATBOOST:
        models['catboost'] = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=False
        )

    return models


def get_optimized_models():
    """
    Get dictionary of models with optimized hyperparameters.

    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            C=0.1,
            solver='lbfgs',
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        ),
    }

    if HAS_XGBOOST:
        models['xgboost'] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

    if HAS_LIGHTGBM:
        models['lightgbm'] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

    if HAS_CATBOOST:
        models['catboost'] = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        )

    return models


def train_model(model, X_train, y_train):
    """
    Train a single model.

    Args:
        model: Model instance
        X_train: Training features
        y_train: Training target

    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, cv=5):
    """
    Evaluate model using cross-validation and various metrics.

    Args:
        model: Trained model
        X: Features
        y: Target
        cv: Number of cross-validation folds

    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y, y_pred, average='binary', zero_division=0),
    }

    # Cross-validation scores - adjust cv for small datasets
    min_class_count = min(y.value_counts()) if hasattr(y, 'value_counts') else min(np.bincount(y.astype(int)))
    actual_cv = min(cv, min_class_count, len(y))
    if actual_cv >= 2:
        try:
            cv_scores = cross_val_score(model, X, y, cv=actual_cv, scoring='accuracy')
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
        except ValueError:
            metrics['cv_accuracy_mean'] = metrics['accuracy']
            metrics['cv_accuracy_std'] = 0.0
    else:
        metrics['cv_accuracy_mean'] = metrics['accuracy']
        metrics['cv_accuracy_std'] = 0.0

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_prob)

    return metrics


def train_and_evaluate_all(X_train, y_train, X_val=None, y_val=None, use_optimized=True):
    """
    Train and evaluate all available models.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        use_optimized: Whether to use optimized hyperparameters

    Returns:
        Dictionary mapping model names to (model, metrics) tuples
    """
    models = get_optimized_models() if use_optimized else get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        trained_model = train_model(model, X_train, y_train)

        # Evaluate on validation set if provided, else use training set
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train

        metrics = evaluate_model(trained_model, eval_X, eval_y)
        results[name] = (trained_model, metrics)
        print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    return results


def get_best_model(results, metric='f1'):
    """
    Get the best performing model based on a specified metric.

    Args:
        results: Dictionary from train_and_evaluate_all
        metric: Metric to use for comparison

    Returns:
        Tuple of (model_name, model, metrics)
    """
    best_name = None
    best_score = -np.inf
    best_model = None
    best_metrics = None

    for name, (model, metrics) in results.items():
        score = metrics.get(metric, 0)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
            best_metrics = metrics

    return best_name, best_model, best_metrics
