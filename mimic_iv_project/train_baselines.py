from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig, TASK_COLUMNS
from .metrics import binary_auprc, binary_auroc, binary_brier


def _load_optional_baseline_deps():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for baseline training. Install with "
            "`python3 -m pip install --user scikit-learn xgboost`."
        ) from exc

    try:
        import xgboost as xgb
    except ImportError:
        xgb = None

    return LogisticRegression, RandomForestClassifier, xgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular baselines on MIMIC-IV features.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else ProjectConfig().project_root
    config = ProjectConfig(project_root=project_root)
    LogisticRegression, RandomForestClassifier, xgb = _load_optional_baseline_deps()

    tabular = pd.read_csv(config.tabular_features_path)
    with open(config.tabular_metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_columns = metadata["feature_columns"]
    x_train = tabular.loc[tabular["split"] == "train", feature_columns].to_numpy(dtype=np.float32)
    x_val = tabular.loc[tabular["split"] == "val", feature_columns].to_numpy(dtype=np.float32)
    x_test = tabular.loc[tabular["split"] == "test", feature_columns].to_numpy(dtype=np.float32)

    model_specs = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=args.n_jobs),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=config.random_seed,
            n_jobs=args.n_jobs,
            class_weight="balanced",
        ),
    }
    if xgb is not None:
        model_specs["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=args.n_jobs,
        )

    results = {}
    for task in TASK_COLUMNS:
        y_train = tabular.loc[tabular["split"] == "train", task].to_numpy(dtype=np.int64)
        y_val = tabular.loc[tabular["split"] == "val", task].to_numpy(dtype=np.int64)
        y_test = tabular.loc[tabular["split"] == "test", task].to_numpy(dtype=np.int64)
        task_results = {}

        if np.unique(y_train).size < 2:
            results[task] = {
                "skipped": "train split contains a single class in the current dataset"
            }
            continue

        for model_name, factory in model_specs.items():
            model = factory()
            model.fit(x_train, y_train)
            val_prob = model.predict_proba(x_val)[:, 1]
            test_prob = model.predict_proba(x_test)[:, 1]
            task_results[model_name] = {
                "val_auroc": binary_auroc(y_val, val_prob),
                "val_auprc": binary_auprc(y_val, val_prob),
                "test_auroc": binary_auroc(y_test, test_prob),
                "test_auprc": binary_auprc(y_test, test_prob),
                "test_brier": binary_brier(y_test, test_prob),
            }

        results[task] = task_results

    output_path = config.artifacts_dir / "baseline_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
