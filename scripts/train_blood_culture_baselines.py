from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_iv_project.config import ProjectConfig
from mimic_iv_project.metrics import binary_auprc, binary_auroc, binary_brier


def _load_ml_deps():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required. Install with `python3 -m pip install --user scikit-learn xgboost`."
        ) from exc

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required. Install with `python3 -m pip install --user xgboost`."
        ) from exc

    return {
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "xgb": xgb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression and XGBoost baselines for blood-culture labels."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    return parser.parse_args()


def _subject_split(subjects: np.ndarray, seed: int) -> tuple[set[int], set[int], set[int]]:
    rng = np.random.default_rng(seed)
    subjects = subjects.copy()
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = set(subjects[:n_train].tolist())
    val = set(subjects[n_train:n_train + n_val].tolist())
    test = set(subjects[n_train + n_val:].tolist())
    return train, val, test


def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def _classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    acc_fn,
    f1_fn,
    precision_fn,
    recall_fn,
) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "threshold": float(threshold),
        "f1": float(f1_fn(y_true, y_pred)),
        "precision": float(precision_fn(y_true, y_pred, zero_division=0)),
        "recall": float(recall_fn(y_true, y_pred, zero_division=0)),
        "accuracy": float(acc_fn(y_true, y_pred)),
        "auroc": binary_auroc(y_true, y_prob),
        "auprc": binary_auprc(y_true, y_prob),
        "brier": binary_brier(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }


def main() -> None:
    args = parse_args()
    deps = _load_ml_deps()
    project_root = args.project_root.resolve()
    config = ProjectConfig(project_root=project_root)
    out_dir = args.out_dir or (config.artifacts_dir / "blood_culture")
    features_path = out_dir / "first_gp_alert_features.csv"
    metadata_path = out_dir / "blood_culture_feature_metadata.json"

    features = pd.read_csv(features_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    data = features[features["is_high_confidence_binary"] == 1].copy()
    data["target_true_bsi"] = data["target_true_bsi"].astype(int)

    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=args.random_seed,
    )
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    feature_columns = metadata["feature_columns"]
    organism_feature_columns = metadata["organism_feature_columns"]
    feature_sets = {
        "full_features": feature_columns,
        "no_organism_features": [col for col in feature_columns if col not in organism_feature_columns],
    }

    x_train_full = data.loc[data["split"] == "train", feature_columns]
    numeric_columns = x_train_full.columns.tolist()

    preprocess = deps["ColumnTransformer"](
        transformers=[
            (
                "num",
                deps["Pipeline"](
                    steps=[
                        ("imputer", deps["SimpleImputer"](strategy="median")),
                        ("scaler", deps["StandardScaler"]()),
                    ]
                ),
                numeric_columns,
            )
        ],
        remainder="drop",
    )

    y_train = data.loc[data["split"] == "train", "target_true_bsi"].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", "target_true_bsi"].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", "target_true_bsi"].to_numpy(dtype=int)
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    results: dict[str, object] = {
        "cohort": {
            "rows": int(len(data)),
            "train_rows": int((data["split"] == "train").sum()),
            "val_rows": int((data["split"] == "val").sum()),
            "test_rows": int((data["split"] == "test").sum()),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "test_positive": int(y_test.sum()),
        },
        "models": {},
    }

    for feature_set_name, columns in feature_sets.items():
        x_train = data.loc[data["split"] == "train", columns]
        x_val = data.loc[data["split"] == "val", columns]
        x_test = data.loc[data["split"] == "test", columns]

        logistic = deps["Pipeline"](
            steps=[
                (
                    "prep",
                    deps["ColumnTransformer"](
                        transformers=[
                            (
                                "num",
                                deps["Pipeline"](
                                    steps=[
                                        ("imputer", deps["SimpleImputer"](strategy="median")),
                                        ("scaler", deps["StandardScaler"]()),
                                    ]
                                ),
                                columns,
                            )
                        ],
                        remainder="drop",
                    ),
                ),
                ("model", deps["LogisticRegression"](max_iter=2000, class_weight="balanced")),
            ]
        )
        logistic.fit(x_train, y_train)
        val_prob = logistic.predict_proba(x_val)[:, 1]
        threshold, best_val_f1 = _best_threshold_by_f1(y_val, val_prob)
        test_prob = logistic.predict_proba(x_test)[:, 1]

        feature_result = {
            "logistic_regression": {
                "validation": _classification_metrics(
                    y_val,
                    val_prob,
                    threshold,
                    deps["accuracy_score"],
                    deps["f1_score"],
                    deps["precision_score"],
                    deps["recall_score"],
                ),
                "test": _classification_metrics(
                    y_test,
                    test_prob,
                    threshold,
                    deps["accuracy_score"],
                    deps["f1_score"],
                    deps["precision_score"],
                    deps["recall_score"],
                ),
            }
        }
        feature_result["logistic_regression"]["validation"]["best_f1_scan"] = best_val_f1

        xgb_model = deps["xgb"].XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=args.random_seed,
        )
        xgb_model.fit(x_train, y_train)
        val_prob = xgb_model.predict_proba(x_val)[:, 1]
        threshold, best_val_f1 = _best_threshold_by_f1(y_val, val_prob)
        test_prob = xgb_model.predict_proba(x_test)[:, 1]
        feature_result["xgboost"] = {
            "validation": _classification_metrics(
                y_val,
                val_prob,
                threshold,
                deps["accuracy_score"],
                deps["f1_score"],
                deps["precision_score"],
                deps["recall_score"],
            ),
            "test": _classification_metrics(
                y_test,
                test_prob,
                threshold,
                deps["accuracy_score"],
                deps["f1_score"],
                deps["precision_score"],
                deps["recall_score"],
            ),
        }
        feature_result["xgboost"]["validation"]["best_f1_scan"] = best_val_f1
        results["models"][feature_set_name] = feature_result

    metrics_path = project_root / "reports" / "blood_culture_baseline_metrics.json"
    split_path = project_root / "reports" / "blood_culture_baseline_split_counts.json"
    split_counts = {
        "train_subjects": int(len(train_subjects)),
        "val_subjects": int(len(val_subjects)),
        "test_subjects": int(len(test_subjects)),
    }
    metrics_path.write_text(json.dumps(results, indent=2))
    split_path.write_text(json.dumps(split_counts, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
