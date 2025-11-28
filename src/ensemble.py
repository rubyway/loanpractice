"""
Ensemble methods for combining multiple models.
"""

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def create_voting_ensemble(models_dict, voting='soft'):
    """
    Create a voting ensemble from multiple models.

    Args:
        models_dict: Dictionary mapping model names to model instances
        voting: 'soft' for probability averaging, 'hard' for majority vote

    Returns:
        VotingClassifier instance
    """
    estimators = [(name, model) for name, model in models_dict.items()]
    ensemble = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
    return ensemble


def create_stacking_ensemble(models_dict, final_estimator=None):
    """
    Create a stacking ensemble from multiple models.

    Args:
        models_dict: Dictionary mapping model names to model instances
        final_estimator: Final estimator for stacking (default: LogisticRegression)

    Returns:
        StackingClassifier instance
    """
    estimators = [(name, model) for name, model in models_dict.items()]

    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=True  # Include original features in final estimator
    )

    return ensemble


def weighted_average_predictions(predictions_list, weights=None):
    """
    Combine predictions using weighted average.

    Args:
        predictions_list: List of prediction arrays (probabilities)
        weights: Optional weights for each model (default: equal weights)

    Returns:
        Weighted average predictions
    """
    predictions = np.array(predictions_list)

    if weights is None:
        weights = np.ones(len(predictions_list)) / len(predictions_list)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

    weighted_preds = np.average(predictions, axis=0, weights=weights)
    return weighted_preds


def optimize_ensemble_weights(predictions_list, y_true, metric='f1'):
    """
    Find optimal weights for ensemble predictions using grid search.

    Args:
        predictions_list: List of prediction probability arrays
        y_true: True labels
        metric: Metric to optimize ('f1', 'accuracy', 'roc_auc')

    Returns:
        Optimal weights array
    """
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    n_models = len(predictions_list)
    best_weights = np.ones(n_models) / n_models
    best_score = -np.inf

    # Generate weight combinations
    weight_grid = np.linspace(0.1, 1.0, 5)

    # Simple grid search for small number of models
    if n_models <= 3:
        for w in np.ndindex(*([5] * n_models)):
            weights = np.array([weight_grid[i] for i in w])
            if weights.sum() > 0:
                weights = weights / weights.sum()

                avg_pred = weighted_average_predictions(predictions_list, weights)
                y_pred = (avg_pred > 0.5).astype(int)

                if metric == 'f1':
                    score = f1_score(y_true, y_pred)
                elif metric == 'accuracy':
                    score = accuracy_score(y_true, y_pred)
                elif metric == 'roc_auc':
                    score = roc_auc_score(y_true, avg_pred)
                else:
                    score = f1_score(y_true, y_pred)

                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
    else:
        # For more models, use random search
        for _ in range(1000):
            weights = np.random.rand(n_models)
            weights = weights / weights.sum()

            avg_pred = weighted_average_predictions(predictions_list, weights)
            y_pred = (avg_pred > 0.5).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'roc_auc':
                score = roc_auc_score(y_true, avg_pred)
            else:
                score = f1_score(y_true, y_pred)

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

    return best_weights


def train_ensemble(X_train, y_train, X_val=None, y_val=None, ensemble_type='voting'):
    """
    Train an ensemble model.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        ensemble_type: 'voting' or 'stacking'

    Returns:
        Trained ensemble and metrics
    """
    from .models import get_optimized_models

    base_models = get_optimized_models()

    # Remove models that might not work well with stacking
    base_models_filtered = {k: v for k, v in base_models.items()
                           if k not in ['svm']}  # SVM can be slow

    if ensemble_type == 'voting':
        ensemble = create_voting_ensemble(base_models_filtered)
    else:
        ensemble = create_stacking_ensemble(base_models_filtered)

    print(f"Training {ensemble_type} ensemble...")
    ensemble.fit(X_train, y_train)

    # Evaluate
    eval_X = X_val if X_val is not None else X_train
    eval_y = y_val if y_val is not None else y_train

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_pred = ensemble.predict(eval_X)
    y_prob = ensemble.predict_proba(eval_X)[:, 1]

    metrics = {
        'accuracy': accuracy_score(eval_y, y_pred),
        'f1': f1_score(eval_y, y_pred),
        'roc_auc': roc_auc_score(eval_y, y_prob)
    }

    print(f"Ensemble {ensemble_type}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    return ensemble, metrics
