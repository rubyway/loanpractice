#!/usr/bin/env python
"""Run experiments and log validation metrics for each model configuration."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.data_preprocessing import get_train_val_split, load_data, preprocess_data
from src.deep_learning import (
    HAS_TORCH,
    build_nn_model,
    build_transformer_nn_model,
    evaluate_nn_model,
    train_nn_model,
)
from src.models import get_best_model, train_and_evaluate_all


@dataclass
class RunMetadata:
    run_id: str
    timestamp: str
    train_path: str
    test_path: str
    val_size: float
    use_nn: bool
    nn_models: List[str]
    ensemble: str
    notes: str
    extra_params: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute loan prediction run and log metrics.")
    parser.add_argument("--train", required=True, help="Path to training CSV file")
    parser.add_argument("--test", required=True, help="Path to test CSV file")
    parser.add_argument("--original", help="Path to original CSV file")
    parser.add_argument("--log-file", default="run_history.csv", help="CSV file to append run metrics")
    parser.add_argument("--run-id", help="Optional run identifier (defaults to timestamp)")
    parser.add_argument("--notes", default="", help="Optional notes to store with the run")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--use-nn", action="store_true", help="Include neural networks in the run")
    parser.add_argument(
        "--nn-models",
        nargs="+",
        default=["feedforward"],
        choices=["feedforward", "transformer"],
        help="Neural net architectures to train when --use-nn is set",
    )
    parser.add_argument("--ffn-variant", choices=["double", "gated"], default="double",
                        help="Feed-forward variant for transformer blocks")
    parser.add_argument("--ensemble", choices=["none", "voting", "stacking", "weighted"],
                        default="none", help="Record desired ensemble strategy (not executed here)")
    return parser.parse_args()


def _generate_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _train_neural_networks(
    args: argparse.Namespace,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Tuple[object, Dict[str, float]]]:
    results: Dict[str, Tuple[object, Dict[str, float]]] = {}
    if not args.use_nn:
        return results
    if not HAS_TORCH:
        print("PyTorch not available; skipping neural networks")
        return results

    nn_builders = {
        "feedforward": build_nn_model,
        "transformer": lambda input_dim: build_transformer_nn_model(
            input_dim=input_dim, ffn_variant=args.ffn_variant
        ),
    }

    for nn_type in dict.fromkeys(args.nn_models):
        builder = nn_builders[nn_type]
        nn_name = f"nn_{nn_type}"
        print(f"Training {nn_name}...")
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
        metrics = evaluate_nn_model(trained_model, X_val.values, y_val.values)
        results[nn_name] = (trained_model, metrics)
        print(f"  {nn_name} F1={metrics.get('f1', 0):.4f}")
    return results


def _append_log_rows(
    log_file: Path,
    metadata: RunMetadata,
    results: Dict[str, Tuple[object, Dict[str, float]]],
) -> None:
    fieldnames = [
        "run_id",
        "timestamp",
        "train_path",
        "test_path",
        "model_name",
        "config_json",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "notes",
    ]

    config_payload = {
        "val_size": metadata.val_size,
        "use_nn": metadata.use_nn,
        "nn_models": metadata.nn_models,
        "ensemble": metadata.ensemble,
        **metadata.extra_params,
    }

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_file.exists()
    with log_file.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for model_name, (_, metrics) in results.items():
            writer.writerow(
                {
                    "run_id": metadata.run_id,
                    "timestamp": metadata.timestamp,
                    "train_path": metadata.train_path,
                    "test_path": metadata.test_path,
                    "model_name": model_name,
                    "config_json": json.dumps(config_payload, ensure_ascii=True),
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "notes": metadata.notes,
                }
            )
    print(f"Logged {len(results)} model entries to {log_file}")


def main() -> None:
    args = parse_args()
    run_id = _generate_run_id(args.run_id)
    timestamp = datetime.now(timezone.utc).isoformat()

    print("Loading data...")
    train_df, test_df, _ = load_data(args.train, args.test, args.original)

    print("Preprocessing data...")
    X_train, y_train, _, *_ = preprocess_data(train_df, test_df)

    print("Splitting validation data...")
    X_tr, X_val, y_tr, y_val = get_train_val_split(X_train, y_train, val_size=args.val_size)

    print("Training baseline models...")
    results = train_and_evaluate_all(X_tr, y_tr, X_val, y_val, use_optimized=True)

    print("Evaluating neural networks (if requested)...")
    nn_results = _train_neural_networks(args, X_tr, y_tr, X_val, y_val)
    results.update(nn_results)

    best_name, _, best_metrics = get_best_model(results, metric="f1")
    print(f"Best model this run: {best_name} (F1={best_metrics.get('f1', 0):.4f})")

    metadata = RunMetadata(
        run_id=run_id,
        timestamp=timestamp,
        train_path=args.train,
        test_path=args.test,
        val_size=args.val_size,
        use_nn=args.use_nn,
        nn_models=args.nn_models if args.use_nn else [],
        ensemble=args.ensemble,
        notes=args.notes,
        extra_params={"ffn_variant": args.ffn_variant},
    )

    log_file = Path(args.log_file)
    _append_log_rows(log_file, metadata, results)


if __name__ == "__main__":
    main()
