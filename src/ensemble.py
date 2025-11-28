"""
Ensemble methods for combining multiple models.
用于组合多个模型的集成方法。
"""

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def create_voting_ensemble(models_dict, voting='soft'):
    """
    Create a voting ensemble from multiple models.
    从多个模型创建投票集成。

    Args / 参数:
        models_dict: Dictionary mapping model names to model instances / 模型名称到模型实例的字典映射
        voting: 'soft' for probability averaging, 'hard' for majority vote / 'soft' 为概率平均，'hard' 为多数投票

    Returns / 返回:
        VotingClassifier instance / VotingClassifier 实例
    """
    estimators = [(name, model) for name, model in models_dict.items()]
    ensemble = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
    return ensemble


def create_stacking_ensemble(models_dict, final_estimator=None):
    """
    Create a stacking ensemble from multiple models.
    从多个模型创建堆叠集成。

    Args / 参数:
        models_dict: Dictionary mapping model names to model instances / 模型名称到模型实例的字典映射
        final_estimator: Final estimator for stacking (default: LogisticRegression) / 用于堆叠的最终估计器（默认：LogisticRegression）

    Returns / 返回:
        StackingClassifier instance / StackingClassifier 实例
    """
    estimators = [(name, model) for name, model in models_dict.items()]

    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=True  # Include original features in final estimator / 在最终估计器中包含原始特征
    )

    return ensemble


def weighted_average_predictions(predictions_list, weights=None):
    """
    Combine predictions using weighted average.
    使用加权平均组合预测。

    Args / 参数:
        predictions_list: List of prediction arrays (probabilities) / 预测数组列表（概率）
        weights: Optional weights for each model (default: equal weights) / 可选的每个模型的权重（默认：相等权重）

    Returns / 返回:
        Weighted average predictions / 加权平均预测
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
    使用网格搜索找到集成预测的最优权重。

    Args / 参数:
        predictions_list: List of prediction probability arrays / 预测概率数组列表
        y_true: True labels / 真实标签
        metric: Metric to optimize ('f1', 'accuracy', 'roc_auc') / 要优化的指标（'f1'、'accuracy'、'roc_auc'）

    Returns / 返回:
        Optimal weights array / 最优权重数组
    """
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    n_models = len(predictions_list)
    best_weights = np.ones(n_models) / n_models
    best_score = -np.inf

    # Generate weight combinations / 生成权重组合
    weight_grid = np.linspace(0.1, 1.0, 5)

    # Simple grid search for small number of models / 针对少量模型的简单网格搜索
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
        # For more models, use random search / 针对更多模型，使用随机搜索
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
    训练集成模型。

    Args / 参数:
        X_train: Training features / 训练特征
        y_train: Training target / 训练目标
        X_val: Validation features / 验证特征
        y_val: Validation target / 验证目标
        ensemble_type: 'voting' or 'stacking' / 'voting' 或 'stacking'

    Returns / 返回:
        Trained ensemble and metrics / 已训练的集成模型和指标
    """
    from .models import get_optimized_models

    base_models = get_optimized_models()

    # Remove models that might not work well with stacking / 移除可能不适合堆叠的模型
    base_models_filtered = {k: v for k, v in base_models.items()
                           if k not in ['svm']}  # SVM can be slow / SVM 可能较慢

    if ensemble_type == 'voting':
        ensemble = create_voting_ensemble(base_models_filtered)
    else:
        ensemble = create_stacking_ensemble(base_models_filtered)

    print(f"Training {ensemble_type} ensemble...")
    ensemble.fit(X_train, y_train)

    # Evaluate / 评估
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
