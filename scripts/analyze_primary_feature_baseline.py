from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRIMARY_FEATURES = [
    "anchor_age",
    "in_icu_at_alert",
    "vasopressor_event_count_24h",
    "vasopressor_active_24h",
    "vasopressor_on_at_alert",
    "mechanical_ventilation_chart_events_24h",
    "mechanical_ventilation_24h",
    "prior_positive_specimens_24h",
    "prior_positive_specimens_7d",
    "prior_gp_positive_specimens_24h",
    "prior_gp_positive_specimens_7d",
    "lab_wbc_last_24h",
    "lab_wbc_min_24h",
    "lab_wbc_max_24h",
    "lab_wbc_count_24h",
    "lab_platelets_last_24h",
    "lab_platelets_min_24h",
    "lab_platelets_count_24h",
    "lab_creatinine_last_24h",
    "lab_creatinine_max_24h",
    "lab_creatinine_count_24h",
    "lab_lactate_last_24h",
    "lab_lactate_max_24h",
    "lab_lactate_count_24h",
    "vital_heart_rate_last_24h",
    "vital_heart_rate_min_24h",
    "vital_heart_rate_max_24h",
    "vital_heart_rate_count_24h",
    "vital_resp_rate_last_24h",
    "vital_resp_rate_max_24h",
    "vital_resp_rate_count_24h",
    "vital_temperature_c_last_24h",
    "vital_temperature_c_min_24h",
    "vital_temperature_c_max_24h",
    "vital_temperature_c_count_24h",
    "vital_map_last_24h",
    "vital_map_min_24h",
    "vital_map_count_24h",
    "vital_spo2_last_24h",
    "vital_spo2_min_24h",
    "vital_spo2_count_24h",
]


def _subject_split(subjects: np.ndarray, seed: int = 7) -> tuple[set[int], set[int], set[int]]:
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(subjects), dtype=int)
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
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _binary_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _binary_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def _binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true.astype(float)) ** 2))


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": _binary_auroc(y_true, y_prob),
        "auprc": _binary_auprc(y_true, y_prob),
        "brier": _binary_brier(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
    }


def _save_barplot(
    data: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    path: Path,
    color: str,
    n: int = 15,
) -> None:
    plot_df = data.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df[label_col], plot_df[value_col], color=color)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_heatmap(corr: pd.DataFrame, path: Path) -> None:
    values = corr.to_numpy()
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    masked_values = np.ma.array(values, mask=mask)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(masked_values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Primary Feature Correlation Matrix (Non-constant Features)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    project_root = Path.cwd()
    features_path = project_root / "artifacts" / "blood_culture" / "first_gp_alert_features.csv"
    reports_dir = project_root / "reports"
    figures_dir = project_root / "figures" / "primary_baseline"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)
    data = df[df["is_high_confidence_binary"] == 1].copy()
    data["target_true_bsi"] = data["target_true_bsi"].astype(int)

    train_subjects, val_subjects, test_subjects = _subject_split(
        data["subject_id"].drop_duplicates().astype(int).to_numpy(),
        seed=7,
    )
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(val_subjects), "val", "test"),
    )

    X_train = data.loc[data["split"] == "train", PRIMARY_FEATURES]
    X_val = data.loc[data["split"] == "val", PRIMARY_FEATURES]
    X_test = data.loc[data["split"] == "test", PRIMARY_FEATURES]
    y_train = data.loc[data["split"] == "train", "target_true_bsi"].to_numpy(dtype=int)
    y_val = data.loc[data["split"] == "val", "target_true_bsi"].to_numpy(dtype=int)
    y_test = data.loc[data["split"] == "test", "target_true_bsi"].to_numpy(dtype=int)

    logistic = Pipeline(
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
                            PRIMARY_FEATURES,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    logistic.fit(X_train, y_train)
    val_prob_lr = logistic.predict_proba(X_val)[:, 1]
    threshold_lr, best_f1_lr = _best_threshold_by_f1(y_val, val_prob_lr)
    test_prob_lr = logistic.predict_proba(X_test)[:, 1]

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
    xgb_model.fit(X_train, y_train)
    val_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
    threshold_xgb, best_f1_xgb = _best_threshold_by_f1(y_val, val_prob_xgb)
    test_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    logistic_coefs = pd.DataFrame(
        {
            "feature": PRIMARY_FEATURES,
            "coefficient": logistic.named_steps["model"].coef_[0],
        }
    )
    logistic_coefs["abs_coefficient"] = logistic_coefs["coefficient"].abs()
    logistic_coefs = logistic_coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    dtest = xgb.DMatrix(X_test[PRIMARY_FEATURES], feature_names=PRIMARY_FEATURES)
    shap_values = xgb_model.get_booster().predict(dtest, pred_contribs=True)
    shap_importance = pd.DataFrame(
        {
            "feature": PRIMARY_FEATURES,
            "mean_abs_shap": np.abs(shap_values[:, :-1]).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    feature_frame = data[PRIMARY_FEATURES].copy()
    feature_stats = pd.DataFrame(
        {
            "feature": PRIMARY_FEATURES,
            "non_missing_count": [int(feature_frame[col].notna().sum()) for col in PRIMARY_FEATURES],
            "observation_rate": [float(feature_frame[col].notna().mean()) for col in PRIMARY_FEATURES],
            "unique_non_missing": [int(feature_frame[col].nunique(dropna=True)) for col in PRIMARY_FEATURES],
            "variance": [float(feature_frame[col].var(skipna=True)) for col in PRIMARY_FEATURES],
        }
    )
    nonconstant_features = feature_stats.loc[feature_stats["unique_non_missing"] > 1, "feature"].tolist()
    dropped_constant_features = feature_stats.loc[feature_stats["unique_non_missing"] <= 1, "feature"].tolist()

    corr = feature_frame[nonconstant_features].corr()
    corr_pairs = []
    for i, left in enumerate(nonconstant_features):
        for right in nonconstant_features[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            corr_pairs.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "correlation": float(value),
                    "abs_correlation": float(abs(value)),
                }
            )
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    logistic_coefs.to_csv(reports_dir / "blood_culture_primary_logistic_coefficients.csv", index=False)
    shap_importance.to_csv(reports_dir / "blood_culture_primary_xgb_shap_importance.csv", index=False)
    feature_stats.sort_values(["observation_rate", "feature"]).to_csv(
        reports_dir / "blood_culture_primary_feature_observation_rates.csv",
        index=False,
    )
    corr.to_csv(reports_dir / "blood_culture_primary_feature_correlation_matrix.csv", index=True)
    corr_pairs_df.to_csv(reports_dir / "blood_culture_primary_feature_correlation_pairs.csv", index=False)

    _save_barplot(
        shap_importance,
        value_col="mean_abs_shap",
        label_col="feature",
        title="XGBoost Mean Absolute SHAP Importance",
        path=reports_dir / "blood_culture_primary_xgb_shap_importance.png",
        color="#2d6a4f",
    )
    _save_barplot(
        logistic_coefs,
        value_col="abs_coefficient",
        label_col="feature",
        title="Logistic Regression Absolute Coefficients",
        path=reports_dir / "blood_culture_primary_logistic_coefficients.png",
        color="#1d3557",
    )
    _save_heatmap(corr, reports_dir / "blood_culture_primary_feature_correlation.png")
    shutil.copy2(reports_dir / "blood_culture_primary_xgb_shap_importance.png", figures_dir / "xgb_shap_importance.png")
    shutil.copy2(
        reports_dir / "blood_culture_primary_logistic_coefficients.png",
        figures_dir / "logistic_coefficients.png",
    )
    shutil.copy2(
        reports_dir / "blood_culture_primary_feature_correlation.png",
        figures_dir / "feature_correlation.png",
    )

    summary = {
        "feature_set_name": "PRIMARY_FEATURES",
        "feature_count": len(PRIMARY_FEATURES),
        "correlation_plot": {
            "features_shown": len(nonconstant_features),
            "dropped_constant_features": dropped_constant_features,
        },
        "cohort": {
            "rows": int(len(data)),
            "unique_patients": int(data["subject_id"].nunique()),
            "unique_admissions": int(data["hadm_id"].nunique()),
            "train_rows": int((data["split"] == "train").sum()),
            "val_rows": int((data["split"] == "val").sum()),
            "test_rows": int((data["split"] == "test").sum()),
            "train_subjects": int(len(train_subjects)),
            "val_subjects": int(len(val_subjects)),
            "test_subjects": int(len(test_subjects)),
        },
        "labels": {
            "positives": int(data["target_true_bsi"].sum()),
            "negatives": int((1 - data["target_true_bsi"]).sum()),
        },
        "models": {
            "logistic_regression": {
                "validation": {**_classification_metrics(y_val, val_prob_lr, threshold_lr), "best_f1_scan": best_f1_lr},
                "test": _classification_metrics(y_test, test_prob_lr, threshold_lr),
            },
            "xgboost": {
                "validation": {**_classification_metrics(y_val, val_prob_xgb, threshold_xgb), "best_f1_scan": best_f1_xgb},
                "test": _classification_metrics(y_test, test_prob_xgb, threshold_xgb),
            },
        },
        "top_xgb_shap_features": shap_importance.head(10).to_dict(orient="records"),
        "top_logistic_features": logistic_coefs.head(10).to_dict(orient="records"),
        "top_correlation_pairs": corr_pairs_df.head(15).to_dict(orient="records"),
        "files": {
            "logistic_coefficients_csv": str(reports_dir / "blood_culture_primary_logistic_coefficients.csv"),
            "xgb_shap_importance_csv": str(reports_dir / "blood_culture_primary_xgb_shap_importance.csv"),
            "feature_observation_rates_csv": str(reports_dir / "blood_culture_primary_feature_observation_rates.csv"),
            "correlation_matrix_csv": str(reports_dir / "blood_culture_primary_feature_correlation_matrix.csv"),
            "correlation_pairs_csv": str(reports_dir / "blood_culture_primary_feature_correlation_pairs.csv"),
            "logistic_coefficients_png": str(reports_dir / "blood_culture_primary_logistic_coefficients.png"),
            "xgb_shap_importance_png": str(reports_dir / "blood_culture_primary_xgb_shap_importance.png"),
            "correlation_png": str(reports_dir / "blood_culture_primary_feature_correlation.png"),
        },
    }
    (reports_dir / "blood_culture_primary_explainability_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
