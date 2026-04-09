from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_s_aureus_same_episode_enriched import (  # noqa: E402
    _best_threshold_by_f1,
    _classification_metrics,
    _load_ml_deps,
    _subject_split,
)


def _build_linear_pipeline(features: list[str], *, penalty: str = "l2", l1_ratio: float | None = None) -> Pipeline:
    if penalty == "elasticnet":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_ratio,
            max_iter=5000,
            class_weight="balanced",
            random_state=7,
        )
    else:
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=7)

    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            features,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", model),
        ]
    )


def _fit_and_score(model, x_train, y_train, x_val, y_val, x_test, y_test, deps) -> dict[str, object]:
    model.fit(x_train, y_train)
    val_prob = model.predict_proba(x_val)[:, 1]
    threshold, best_f1 = _best_threshold_by_f1(y_val, val_prob, deps)
    test_prob = model.predict_proba(x_test)[:, 1]
    return {
        "validation": {**_classification_metrics(y_val, val_prob, threshold, deps), "best_f1_scan": best_f1},
        "test": _classification_metrics(y_test, test_prob, threshold, deps),
    }


def main() -> None:
    project_root = Path.cwd()
    features_path = (
        project_root / "artifacts" / "blood_culture" / "s_aureus_same_episode_primary_urgent_enriched_features.csv"
    )
    pruned_metrics_path = project_root / "reports" / "s_aureus_same_episode_pruned_metrics.json"
    out_json_path = project_root / "reports" / "s_aureus_same_episode_model_comparison.json"
    out_md_path = project_root / "reports" / "s_aureus_same_episode_model_comparison.md"

    deps = _load_ml_deps()
    selected = json.loads(pruned_metrics_path.read_text())["feature_list"]
    data = pd.read_csv(features_path)

    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=7,
    )
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    x_train = data.loc[data["split"] == "train", selected]
    x_val = data.loc[data["split"] == "val", selected]
    x_test = data.loc[data["split"] == "test", selected]
    y_train = data.loc[data["split"] == "train", "target_s_aureus_same_episode"].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", "target_s_aureus_same_episode"].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", "target_s_aureus_same_episode"].to_numpy(dtype=int)

    models: dict[str, object] = {}
    models["logistic_regression"] = _fit_and_score(
        _build_linear_pipeline(selected),
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        deps,
    )
    models["elastic_net_logistic"] = _fit_and_score(
        _build_linear_pipeline(selected, penalty="elasticnet", l1_ratio=0.5),
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        deps,
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=7,
        n_jobs=8,
    )
    models["random_forest"] = _fit_and_score(rf, x_train, y_train, x_val, y_val, x_test, y_test, deps)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=7,
        n_jobs=8,
    )
    models["xgboost"] = _fit_and_score(xgb_model, x_train, y_train, x_val, y_val, x_test, y_test, deps)

    results = {
        "task_name": "s_aureus_same_episode_model_comparison",
        "feature_set_name": "PRUNED_ENRICHED_FEATURES",
        "feature_count": len(selected),
        "feature_list": selected,
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
            "positive_rows": int(data["target_s_aureus_same_episode"].sum()),
            "positive_prevalence": float(data["target_s_aureus_same_episode"].mean()),
            "train_rows": int((data["split"] == "train").sum()),
            "val_rows": int((data["split"] == "val").sum()),
            "test_rows": int((data["split"] == "test").sum()),
        },
        "models": models,
    }
    out_json_path.write_text(json.dumps(results, indent=2))

    test_table = pd.DataFrame(
        [
            {
                "model": name,
                "auroc": metrics["test"]["auroc"],
                "auprc": metrics["test"]["auprc"],
                "f1": metrics["test"]["f1"],
                "precision": metrics["test"]["precision"],
                "recall": metrics["test"]["recall"],
                "accuracy": metrics["test"]["accuracy"],
                "brier": metrics["test"]["brier"],
            }
            for name, metrics in models.items()
        ]
    ).sort_values(["auroc", "auprc", "f1"], ascending=False)

    lines = [
        "# Same-Episode `S. aureus` Model Comparison",
        "",
        "This comparison keeps the cohort, split, and feature set fixed.",
        "",
        "- cohort: primary urgent/emergency same-episode first Gram-positive alerts",
        f"- rows: `{results['cohort']['rows']}`",
        f"- unique patients: `{results['cohort']['unique_patients']}`",
        f"- positives: `{results['cohort']['positive_rows']}` (`{results['cohort']['positive_prevalence'] * 100:.1f}%`)",
        f"- features: `{results['feature_count']}` pruned enriched features",
        "",
        "## Held-Out Test Results",
        "",
        "| Model | AUROC | AUPRC | F1 | Precision | Recall | Accuracy | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_table.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['auroc']:.3f} | {row['auprc']:.3f} | {row['f1']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['accuracy']:.3f} | {row['brier']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Logistic Regression remains a strong transparent baseline.",
            "- Elastic Net shows whether a sparser linear model can keep the signal.",
            "- Random Forest checks whether a simpler bagged tree model is enough.",
            "- XGBoost remains the strongest nonlinear benchmark in this repo.",
        ]
    )
    out_md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
