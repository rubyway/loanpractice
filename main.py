#!/usr/bin/env python
"""
Main script for loan prediction using multiple ML and DL methods.
使用多种机器学习和深度学习方法进行贷款预测的主脚本。

Usage / 使用方法:
    python main.py --train data/train.csv --test data/test.csv [--original data/original.csv]
"""

import argparse
import os
import sys
import warnings
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output / 抑制警告以获得更清晰的输出
warnings.filterwarnings('ignore')

# Add src to path / 将 src 添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_data, preprocess_data, get_train_val_split
from src.models import train_and_evaluate_all, get_best_model
from src.ensemble import train_ensemble, weighted_average_predictions, optimize_ensemble_weights
from src.deep_learning import (
    HAS_TORCH,
    build_nn_model,
    build_transformer_nn_model,
    evaluate_nn_model,
    predict_nn_model,
    train_nn_model,
)


def parse_args():
    """
    Parse command line arguments.
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='Loan Prediction Model Training')
    parser.add_argument('--train', required=True, help='Path to training CSV file')
    parser.add_argument('--test', required=True, help='Path to test CSV file')
    parser.add_argument('--original', help='Path to original loan data CSV file (optional)')
    parser.add_argument('--output', default='predictions.csv', help='Output file for predictions')
    parser.add_argument('--use-nn', action='store_true', help='Include neural network models')
    parser.add_argument(
        '--nn-models',
        nargs='+',
        default=['feedforward'],
        choices=['feedforward', 'transformer'],
        help='Neural network architectures to train when --use-nn is set'
    )
    parser.add_argument('--ensemble', choices=['voting', 'stacking', 'weighted', 'none'],
                       default='voting', help='Ensemble method to use')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation set size ratio')
    return parser.parse_args()


def main():
    """
    Main function to run the loan prediction pipeline.
    运行贷款预测流程的主函数。
    """
    args = parse_args()

    print("=" * 60)
    print("Loan Prediction Model Training")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    train_df, test_df, original_df = load_data(args.train, args.test, args.original)
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    if original_df is not None:
        print(f"  Original shape: {original_df.shape}")

    # Preprocess data
    print("\n[2/5] Preprocessing data...")
    X_train, y_train, X_test, encoders, scaler, feature_names, test_index = preprocess_data(
        train_df, test_df
    )
    print(f"  Features: {len(feature_names)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Split into train/validation
    print("\n[3/5] Splitting train/validation...")
    X_tr, X_val, y_tr, y_val = get_train_val_split(X_train, y_train, val_size=args.val_size)
    print(f"  Training set: {len(X_tr)} samples")
    print(f"  Validation set: {len(X_val)} samples")

    # Train and evaluate individual models
    print("\n[4/5] Training and evaluating models...")
    results = train_and_evaluate_all(X_tr, y_tr, X_val, y_val, use_optimized=True)

    nn_result_names = set()

    # Try neural networks if requested
    if args.use_nn:
        if not HAS_TORCH:
            print("  PyTorch not available, skipping neural networks")
        else:
            nn_builders = {
                'feedforward': build_nn_model,
                'transformer': build_transformer_nn_model,
            }
            for nn_type in dict.fromkeys(args.nn_models):
                builder = nn_builders[nn_type]
                nn_name = f"nn_{nn_type}"
                try:
                    print(f"\nTraining Neural Network ({nn_type})...")
                    nn_model = builder(X_tr.shape[1])
                    nn_model, _ = train_nn_model(
                        nn_model,
                        X_tr.values,
                        y_tr.values,
                        X_val.values,
                        y_val.values,
                        epochs=50,
                        batch_size=64,
                    )
                    nn_metrics = evaluate_nn_model(nn_model, X_val.values, y_val.values)
                    results[nn_name] = (nn_model, nn_metrics)
                    nn_result_names.add(nn_name)
                    print(f"  {nn_name}: Accuracy={nn_metrics['accuracy']:.4f}, F1={nn_metrics['f1']:.4f}")
                except Exception as e:
                    print(f"  {nn_type} neural network training failed: {e}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 75)
    for name, (model, metrics) in sorted(results.items(), key=lambda x: -x[1][1].get('f1', 0)):
        print(f"{name:<25} {metrics.get('accuracy', 0):>10.4f} {metrics.get('precision', 0):>10.4f} "
              f"{metrics.get('recall', 0):>10.4f} {metrics.get('f1', 0):>10.4f} "
              f"{metrics.get('roc_auc', 0):>10.4f}")

    # Get best model
    best_name, best_model, best_metrics = get_best_model(results, metric='f1')
    print(f"\nBest individual model: {best_name} (F1={best_metrics['f1']:.4f})")

    # Train ensemble if requested
    print("\n[5/5] Generating predictions...")

    if args.ensemble == 'voting':
        print("Using voting ensemble for final predictions...")
        ensemble_model, ensemble_metrics = train_ensemble(X_tr, y_tr, X_val, y_val, 'voting')
        print(f"Voting Ensemble: Accuracy={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1']:.4f}")

        # Make predictions
        predictions = ensemble_model.predict(X_test)
        probabilities = ensemble_model.predict_proba(X_test)[:, 1]

    elif args.ensemble == 'stacking':
        print("Using stacking ensemble for final predictions...")
        ensemble_model, ensemble_metrics = train_ensemble(X_tr, y_tr, X_val, y_val, 'stacking')
        print(f"Stacking Ensemble: Accuracy={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1']:.4f}")

        # Make predictions
        predictions = ensemble_model.predict(X_test)
        probabilities = ensemble_model.predict_proba(X_test)[:, 1]

    elif args.ensemble == 'weighted':
        print("Using weighted average ensemble...")
        # Get predictions from all models
        predictions_list = []
        model_names = []
        for name, (model, _) in results.items():
            if name in nn_result_names:
                prob = predict_nn_model(model, X_val.values)
            else:
                prob = model.predict_proba(X_val)[:, 1]
            predictions_list.append(prob)
            model_names.append(name)

        # Optimize weights
        optimal_weights = optimize_ensemble_weights(predictions_list, y_val, metric='f1')
        print(f"Optimal weights: {dict(zip(model_names, optimal_weights))}")

        # Get test predictions
        test_predictions_list = []
        for name in model_names:
            model = results[name][0]
            if name in nn_result_names:
                prob = predict_nn_model(model, X_test.values)
            else:
                prob = model.predict_proba(X_test)[:, 1]
            test_predictions_list.append(prob)

        probabilities = weighted_average_predictions(test_predictions_list, optimal_weights)
        predictions = (probabilities > 0.5).astype(int)

    else:  # none - use best model
        print(f"Using best model ({best_name}) for predictions...")
        if best_name in nn_result_names:
            nn_model_for_pred = results[best_name][0]
            probabilities = predict_nn_model(nn_model_for_pred, X_test.values)
            predictions = (probabilities > 0.5).astype(int)
        else:
            predictions = best_model.predict(X_test)
            probabilities = best_model.predict_proba(X_test)[:, 1]

    # Save predictions
    output_df = pd.DataFrame({
        'id': test_index,
        'loan_paid_back': predictions,
        'probability': probabilities
    })
    output_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted loan_paid_back=0: {(predictions == 0).sum()} ({100*(predictions == 0).mean():.1f}%)")
    print(f"Predicted loan_paid_back=1: {(predictions == 1).sum()} ({100*(predictions == 1).mean():.1f}%)")
    print(f"Average probability: {probabilities.mean():.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
