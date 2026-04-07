from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_iv_project.blood_culture import (
    compute_repeat_features,
    deserialize_organisms,
    load_specimen_from_csv,
    serialize_organisms,
)
from mimic_iv_project.config import ProjectConfig


ANTI_MRSA_PATTERNS = {
    "vancomycin_iv_like": re.compile(r"VANCOMYCIN", re.I),
    "linezolid": re.compile(r"LINEZOLID", re.I),
    "daptomycin": re.compile(r"DAPTOMYCIN", re.I),
    "ceftaroline": re.compile(r"CEFTAROLINE", re.I),
}

SYSTEMIC_ABX_PATTERNS = (
    re.compile(
        r"VANCOMYCIN|LINEZOLID|DAPTOMYCIN|CEFTAROLINE|PIPERACILLIN|TAZOBACTAM|ZOSYN|CEFEPIME|MEROPENEM|"
        r"CEFTAZIDIME|IMIPENEM|AZTREONAM|CEFTRIAXONE|CEFAZOLIN|NAFCILLIN|OXACILLIN|AMPICILLIN|"
        r"AMPICILLIN-SULBACTAM|UNASYN|AMOXICILLIN|AUGMENTIN|PENICILLIN|ERTAPENEM|GENTAMICIN|TOBRAMYCIN|"
        r"CIPROFLOXACIN|LEVOFLOXACIN|MOXIFLOXACIN|TRIMETHOPRIM|SULFAMETHOXAZOLE|BACTRIM|CLINDAMYCIN|"
        r"DOXYCYCLINE|METRONIDAZOLE",
        re.I,
    ),
)

MED_EXCLUDE_TOKENS = (
    "ORAL",
    "ENEMA",
    "LOCK",
    "OPHTH",
    "EYE",
    "EAR",
    "OTIC",
    "TOPICAL",
    "CREAM",
    "OINTMENT",
    "GEL",
    "FLUSH",
    "NEB",
    "INHAL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clinically meaningful labels for the first Gram-positive blood-culture alert cohort."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root for artifact output.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Optional MIMIC-IV root containing hosp/ and icu/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <project-root>/artifacts/blood_culture.",
    )
    parser.add_argument(
        "--specimen-path",
        type=Path,
        default=None,
        help="Optional override for positive_blood_culture_specimens.csv.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Optional override for first_gp_alert_cohort.csv.",
    )
    parser.add_argument(
        "--specimen-subset-path",
        type=Path,
        default=None,
        help="Optional override for blood_culture_specimen_subset.csv.",
    )
    return parser.parse_args()


def _episode_specimen_counts_24h(
    cohort: pd.DataFrame,
    specimen_subset: pd.DataFrame,
) -> pd.DataFrame:
    subset = specimen_subset[specimen_subset["hadm_id"].notna()].copy()
    subset["hadm_id"] = subset["hadm_id"].astype("int64")
    subset["history_anchor_time"] = pd.to_datetime(subset["history_anchor_time"], errors="coerce")
    subset = subset.dropna(subset=["history_anchor_time"])

    subset_by_hadm: dict[int, np.ndarray] = {}
    subset_ids_by_hadm: dict[int, np.ndarray] = {}
    for hadm_id, group in subset.groupby("hadm_id", sort=False):
        ordered = group.sort_values(["history_anchor_time", "micro_specimen_id"])
        subset_by_hadm[int(hadm_id)] = ordered["history_anchor_time"].to_numpy(dtype="datetime64[ns]")
        subset_ids_by_hadm[int(hadm_id)] = ordered["micro_specimen_id"].to_numpy(dtype=np.int64)

    episode_count = []
    additional_count = []
    for row in cohort.itertuples(index=False):
        hadm_id = int(row.hadm_id)
        times = subset_by_hadm.get(hadm_id)
        specimen_ids = subset_ids_by_hadm.get(hadm_id)
        if times is None or specimen_ids is None:
            episode_count.append(np.nan)
            additional_count.append(np.nan)
            continue
        anchor = np.datetime64(row.alert_time)
        left = np.searchsorted(times, anchor - np.timedelta64(24, "h"), side="left")
        right = np.searchsorted(times, anchor + np.timedelta64(24, "h"), side="right")
        window_ids = specimen_ids[left:right]
        count = int(len(window_ids))
        extra = int(np.count_nonzero(window_ids != int(row.micro_specimen_id)))
        episode_count.append(count)
        additional_count.append(extra)

    out = cohort.copy()
    out["episode_blood_culture_specimens_24h"] = episode_count
    out["additional_blood_culture_specimens_24h"] = additional_count
    out["multiple_sets_approx_24h"] = (out["additional_blood_culture_specimens_24h"].fillna(0) >= 1).astype(int)
    return out


def _classify_medication(medication: str) -> dict[str, int]:
    med = str(medication or "").upper()
    if not med or any(token in med for token in MED_EXCLUDE_TOKENS):
        return {
            "systemic_abx": 0,
            "anti_mrsa": 0,
        }

    anti_mrsa = int(any(pattern.search(med) for pattern in ANTI_MRSA_PATTERNS.values()))
    systemic = int(any(pattern.search(med) for pattern in SYSTEMIC_ABX_PATTERNS))
    return {
        "systemic_abx": systemic,
        "anti_mrsa": anti_mrsa,
    }


def _post_alert_antibiotic_behavior(
    cohort: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    lookup = cohort[["hadm_id", "alert_time"]].copy()
    lookup["hadm_id"] = lookup["hadm_id"].astype("int64")
    hadm_set = set(lookup["hadm_id"].tolist())

    counts = {
        int(hadm): {
            "systemic_abx_admin_0_24h": 0,
            "systemic_abx_admin_24_72h": 0,
            "anti_mrsa_admin_0_24h": 0,
            "anti_mrsa_admin_24_72h": 0,
        }
        for hadm in hadm_set
    }

    reader = pd.read_csv(
        config.hosp_dir / "emar.csv",
        usecols=["hadm_id", "charttime", "medication", "event_txt"],
        chunksize=500_000,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["hadm_id", "charttime", "medication"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        if chunk.empty:
            continue
        chunk["event_txt"] = chunk["event_txt"].fillna("").astype(str).str.lower()
        chunk = chunk[chunk["event_txt"].str.contains("admin", regex=False)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        if chunk.empty:
            continue
        chunk = chunk.merge(lookup, on="hadm_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] >= chunk["alert_time"])
            & (chunk["charttime"] <= chunk["alert_time"] + pd.Timedelta(hours=72))
        ].copy()
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            flags = _classify_medication(row.medication)
            if not flags["systemic_abx"] and not flags["anti_mrsa"]:
                continue
            delta_hours = (row.charttime - row.alert_time).total_seconds() / 3600.0
            window = "0_24h" if delta_hours < 24 else "24_72h"
            hadm_id = int(row.hadm_id)
            if flags["systemic_abx"]:
                counts[hadm_id][f"systemic_abx_admin_{window}"] += 1
            if flags["anti_mrsa"]:
                counts[hadm_id][f"anti_mrsa_admin_{window}"] += 1

    out = cohort.copy()
    for column in [
        "systemic_abx_admin_0_24h",
        "systemic_abx_admin_24_72h",
        "anti_mrsa_admin_0_24h",
        "anti_mrsa_admin_24_72h",
    ]:
        out[column] = out["hadm_id"].astype(int).map(lambda hadm: counts[int(hadm)][column])
    out["continued_systemic_abx_24_72h"] = (out["systemic_abx_admin_24_72h"] >= 2).astype(int)
    out["continued_anti_mrsa_24_72h"] = (out["anti_mrsa_admin_24_72h"] >= 1).astype(int)
    return out


def _assign_clinical_label(cohort: pd.DataFrame) -> pd.DataFrame:
    labeled = cohort.copy()
    labeled["provisional_label"] = "indeterminate"
    labeled["label_source"] = "fallback_indeterminate"

    clear_pathogen = labeled["category"].eq("true_like")
    repeated_commensal = labeled["category"].eq("contam_like") & labeled["repeat_same_organism_48h"].eq(1)
    microbiology_true = clear_pathogen | repeated_commensal

    continued_treatment = labeled["continued_systemic_abx_24_72h"].eq(1) | labeled["continued_anti_mrsa_24_72h"].eq(1)

    probable_true = microbiology_true & continued_treatment

    strict_contaminant_micro = (
        labeled["category"].eq("contam_like")
        & labeled["repeat_any_positive_48h"].eq(0)
        & labeled["repeat_same_organism_48h"].eq(0)
        & labeled["multiple_sets_approx_24h"].eq(1)
    )
    not_clinically_acted_on = (
        labeled["systemic_abx_admin_24_72h"].eq(0)
        & labeled["anti_mrsa_admin_24_72h"].eq(0)
    )
    probable_contaminant = strict_contaminant_micro & not_clinically_acted_on

    labeled.loc[probable_true, "provisional_label"] = "probable_clinically_significant_bsi_alert"
    labeled.loc[probable_true, "label_source"] = np.where(
        clear_pathogen[probable_true],
        "clear_pathogen_plus_continued_treatment_24_72h",
        "repeated_commensal_plus_continued_treatment_24_72h",
    )
    labeled.loc[probable_contaminant, "provisional_label"] = "probable_contaminant_or_low_significance_alert"
    labeled.loc[probable_contaminant, "label_source"] = "commensal_single_episode_no_repeat_no_treatment_24_72h"
    labeled["is_high_confidence_binary"] = labeled["provisional_label"].isin(
        {
            "probable_clinically_significant_bsi_alert",
            "probable_contaminant_or_low_significance_alert",
        }
    ).astype(int)
    return labeled


def main() -> None:
    args = parse_args()
    config = ProjectConfig(project_root=args.project_root, raw_root=args.raw_root)
    out_dir = args.out_dir or (config.artifacts_dir / "blood_culture")
    out_dir.mkdir(parents=True, exist_ok=True)

    specimen_path = args.specimen_path or (out_dir / "positive_blood_culture_specimens.csv")
    cohort_path = args.cohort_path or (out_dir / "first_gp_alert_cohort.csv")
    specimen_subset_path = args.specimen_subset_path or (out_dir / "blood_culture_specimen_subset.csv")

    specimen = load_specimen_from_csv(specimen_path)
    cohort = pd.read_csv(cohort_path)
    cohort["alert_time"] = pd.to_datetime(cohort["alert_time"], errors="coerce")
    cohort["organisms"] = cohort["organisms_json"].map(deserialize_organisms)

    specimen_subset = pd.read_csv(specimen_subset_path)
    specimen_subset["history_anchor_time"] = pd.to_datetime(specimen_subset["history_anchor_time"], errors="coerce")

    labeled = compute_repeat_features(cohort, specimen)
    labeled = _episode_specimen_counts_24h(labeled, specimen_subset)
    labeled = _post_alert_antibiotic_behavior(labeled, config)
    labeled = _assign_clinical_label(labeled)

    labels_csv = out_dir / "first_gp_alert_labels.csv"
    dataset_csv = out_dir / "first_gp_alert_dataset.csv"
    metadata_json = out_dir / "blood_culture_label_metadata.json"

    label_cols = [
        "hadm_id",
        "micro_specimen_id",
        "repeat_any_positive_48h",
        "repeat_gp_positive_48h",
        "repeat_same_organism_48h",
        "repeat_positive_specimen_count_48h",
        "episode_blood_culture_specimens_24h",
        "additional_blood_culture_specimens_24h",
        "multiple_sets_approx_24h",
        "systemic_abx_admin_0_24h",
        "systemic_abx_admin_24_72h",
        "anti_mrsa_admin_0_24h",
        "anti_mrsa_admin_24_72h",
        "continued_systemic_abx_24_72h",
        "continued_anti_mrsa_24_72h",
        "provisional_label",
        "label_source",
        "is_high_confidence_binary",
    ]
    labeled[label_cols].to_csv(labels_csv, index=False)

    dataset = labeled.copy()
    dataset["organisms_json"] = dataset["organisms"].map(serialize_organisms)
    dataset = dataset.drop(columns=["organisms"])
    dataset.to_csv(dataset_csv, index=False)

    label_counts = labeled["provisional_label"].value_counts().to_dict()
    metadata = {
        "label_definition": {
            "probable_clinically_significant_bsi_alert": (
                "clear pathogen or repeated commensal blood-culture alert with continued antibiotic treatment in the 24-72h post-alert window"
            ),
            "probable_contaminant_or_low_significance_alert": (
                "contaminant-prone organism with no repeat positive blood culture in 48h, at least one additional blood-culture specimen in the 24h episode window, and no post-alert antibiotic continuation in 24-72h"
            ),
            "indeterminate": "all other first Gram-positive alerts",
        },
        "counts": {
            "rows": int(len(labeled)),
            "probable_clinically_significant_bsi_alert": int(
                label_counts.get("probable_clinically_significant_bsi_alert", 0)
            ),
            "probable_contaminant_or_low_significance_alert": int(
                label_counts.get("probable_contaminant_or_low_significance_alert", 0)
            ),
            "indeterminate": int(label_counts.get("indeterminate", 0)),
            "high_confidence_binary": int(labeled["is_high_confidence_binary"].sum()),
        },
        "files": {
            "labels_csv": str(labels_csv),
            "dataset_csv": str(dataset_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
