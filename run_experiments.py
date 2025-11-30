#!/usr/bin/env python
"""Batch experiment runner for predefined commands / 预定义命令批量实验脚本"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data_preprocessing import get_train_val_split, load_data, preprocess_data
from src.deep_learning import (
    HAS_TORCH,
    build_nn_model,
    build_transformer_nn_model,
    predict_nn_model,
    train_nn_model,
)
from src.ensemble import optimize_ensemble_weights, train_ensemble, weighted_average_predictions
from src.models import get_best_model, train_and_evaluate_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preset experiments sequentially and record their validation metrics. "
            "按顺序运行预设实验并记录其验证指标"
        )
    )
    parser.add_argument("--train", required=True, help="Training CSV path / 训练 CSV 路径")
    parser.add_argument("--test", required=True, help="Test CSV path / 测试 CSV 路径")
    parser.add_argument("--original", help="Optional original CSV / 可选原始 CSV")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split ratio (default 0.2) / 验证集占比 (默认 0.2)",
    )
    parser.add_argument(
        "--summary-csv",
        default="logs/experiment_summary.csv",
        help="Where to store metrics log / 保存指标日志的位置",
    )
    return parser.parse_args()


@dataclass
class ExperimentConfig:
    name: str
    description_en: str
    description_zh: str
    ensemble: str
    use_nn: bool = False
    nn_models: Tuple[str, ...] = ("feedforward",)
    ffn_variant: str = "double"


EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="voting_default",
        description_en="Use voting ensemble (default)",
        description_zh="使用投票集成（默认）",
        ensemble="voting",
        use_nn=False,
    ),
    ExperimentConfig(
        name="stacking_ensemble",
        description_en="Use stacking ensemble",
        description_zh="使用堆叠集成",
        ensemble="stacking",
        use_nn=False,
    ),
    ExperimentConfig(
        name="nn_feedforward",
        description_en="Include neural network",
        description_zh="包含神经网络",
        ensemble="voting",
        use_nn=True,
        nn_models=("feedforward",),
        ffn_variant="double",
    ),
    ExperimentConfig(
        name="best_single",
        description_en="Use best single model",
        description_zh="使用最佳单模型",
        ensemble="none",
        use_nn=False,
    ),
    ExperimentConfig(
        name="nn_transformer_gated",
        description_en="Select transformer NN with gated FFN",
        description_zh="使用带 gated FFN 的 Transformer NN",
        ensemble="voting",
        use_nn=True,
        nn_models=("transformer",),
        ffn_variant="gated",
    ),
]


def _prepare_data(args: argparse.Namespace):
    print("Loading and preprocessing data / 加载并预处理数据...")
    train_df, test_df, _ = load_data(args.train, args.test, args.original)
    X_train, y_train, X_test, *_ = preprocess_data(train_df, test_df)
    X_tr, X_val, y_tr, y_val = get_train_val_split(X_train, y_train, val_size=args.val_size)
    return {
        "X_tr": X_tr,
        "X_val": X_val,
        "y_tr": y_tr,
        "y_val": y_val,
        "X_test": X_test,
    }


def _train_neural_networks(
    cfg: ExperimentConfig,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Tuple[object, Dict[str, float]]]:
    results: Dict[str, Tuple[object, Dict[str, float]]] = {}
    if not cfg.use_nn:
        return results
    if not HAS_TORCH:
        print("PyTorch unavailable, skipping NN / 未检测到 PyTorch，跳过神经网络")
        return results

    nn_builders = {
        "feedforward": build_nn_model,
        "transformer": lambda input_dim: build_transformer_nn_model(
            input_dim=input_dim, ffn_variant=cfg.ffn_variant
        ),
    }

    for nn_type in dict.fromkeys(cfg.nn_models):
        builder = nn_builders[nn_type]
        nn_name = f"nn_{nn_type}"
        print(f"Training {nn_name} ... / 训练 {nn_name} ...")
        model = builder(X_tr.shape[1])
        trained_model, _ = train_nn_model(
            model,
            X_tr.values,
            y_tr.values,
            X_val.values,
            y_val.values,
            epochs=50,
            batch_size=64,
        )
        prob = predict_nn_model(trained_model, X_val.values)
        metrics = _compute_metrics(y_val.values, (prob > 0.5).astype(int), prob)
        results[nn_name] = (trained_model, metrics)
    return results


def _compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _predict_probabilities(model, X_val, is_nn: bool):
    if is_nn:
        return predict_nn_model(model, X_val.values)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_val)[:, 1]
    return None


def _run_weighted_ensemble(
    results: Dict[str, Tuple[object, Dict[str, float]]],
    nn_names: set,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    predictions_list = []
    for name, (model, _) in results.items():
        if name in nn_names:
            prob = predict_nn_model(model, X_val.values)
        else:
            prob = model.predict_proba(X_val)[:, 1]
        predictions_list.append(prob)
    weights = optimize_ensemble_weights(predictions_list, y_val, metric="f1")
    combined = weighted_average_predictions(predictions_list, weights)
    preds = (combined > 0.5).astype(int)
    return _compute_metrics(y_val.values, preds, combined)


def run_experiment(
    cfg: ExperimentConfig,
    data_bundle,
):
    X_tr = data_bundle["X_tr"]
    X_val = data_bundle["X_val"]
    y_tr = data_bundle["y_tr"]
    y_val = data_bundle["y_val"]

    print(f"\n=== Running {cfg.name} ===")
    print(f"{cfg.description_en} / {cfg.description_zh}")

    results = train_and_evaluate_all(X_tr, y_tr, X_val, y_val, use_optimized=True)
    nn_results = _train_neural_networks(cfg, X_tr, y_tr, X_val, y_val)
    results.update(nn_results)
    nn_names = set(nn_results.keys())

    final_model = ""
    final_metrics: Dict[str, float]

    if cfg.ensemble in {"voting", "stacking"}:
        ensemble_model, _ = train_ensemble(X_tr, y_tr, X_val, y_val, cfg.ensemble)
        y_pred = ensemble_model.predict(X_val)
        y_prob = ensemble_model.predict_proba(X_val)[:, 1]
        final_metrics = _compute_metrics(y_val.values, y_pred, y_prob)
        final_model = f"ensemble_{cfg.ensemble}"
    elif cfg.ensemble == "weighted":
        final_metrics = _run_weighted_ensemble(results, nn_names, X_val, y_val)
        final_model = "ensemble_weighted"
    elif cfg.ensemble == "none":
        best_name, best_model, _ = get_best_model(results, metric="f1")
        if best_name in nn_names:
            prob = predict_nn_model(best_model, X_val.values)
            pred = (prob > 0.5).astype(int)
        else:
            prob = _predict_probabilities(best_model, X_val, False)
            if prob is None:
                pred = best_model.predict(X_val)
            else:
                pred = (prob > 0.5).astype(int)
        final_metrics = _compute_metrics(y_val.values, pred, prob)
        final_model = best_name
    else:
        raise ValueError(f"Unsupported ensemble type: {cfg.ensemble}")

    return {
        "name": cfg.name,
        "description_en": cfg.description_en,
        "description_zh": cfg.description_zh,
        "ensemble": cfg.ensemble,
        "use_nn": cfg.use_nn,
        "nn_models": ",".join(cfg.nn_models) if cfg.use_nn else "-",
        "ffn_variant": cfg.ffn_variant if cfg.use_nn else "-",
        "final_model": final_model,
        **final_metrics,
    }


def _write_summary(records: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "description_en",
        "description_zh",
        "ensemble",
        "use_nn",
        "nn_models",
        "ffn_variant",
        "final_model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved summary to {csv_path} / 汇总结果已保存至 {csv_path}")


def main() -> None:
    args = parse_args()
    data_bundle = _prepare_data(args)
    records: List[Dict[str, object]] = []

    for cfg in EXPERIMENTS:
        record = run_experiment(cfg, data_bundle)
        records.append(record)
        print(
            f"Result {cfg.name}: F1={record['f1']:.4f}, AUC={record['roc_auc']:.4f}, model={record['final_model']}"
            f" / 结果 {cfg.name}: F1={record['f1']:.4f}, AUC={record['roc_auc']:.4f}, 模型={record['final_model']}"
        )

    best_record = max(records, key=lambda r: r.get("roc_auc") or 0.0)
    print("\n=== Best Experiment / 最佳实验 ===")
    print(
        f"{best_record['name']} ({best_record['description_en']} / {best_record['description_zh']}) "
        f"-> ROC-AUC={best_record['roc_auc']:.4f}, model={best_record['final_model']}"
    )

    _write_summary(records, Path(args.summary_csv))


if __name__ == "__main__":
    main()
