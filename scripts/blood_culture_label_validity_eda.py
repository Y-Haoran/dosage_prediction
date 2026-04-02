from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pandas as pd

from mimic_iv_project.config import ProjectConfig


GRAM_POSITIVE_TERMS = (
    "STAPH",
    "STREP",
    "ENTEROCOCC",
    "CORYNEBACTER",
    "BACILLUS",
    "CUTIBACTER",
    "PROPIONIBACTER",
    "MICROCOCC",
    "LISTERIA",
    "ACTINOMYC",
    "LACTOBACILL",
    "PEPTOSTREP",
    "GEMELLA",
    "AEROCOCC",
    "ROTHIA",
    "LEUCONOSTOC",
)

CONTAMINANT_PATTERNS = (
    "COAGULASE NEGATIVE",
    "STAPHYLOCOCCUS EPIDERMIDIS",
    "STAPHYLOCOCCUS HOMINIS",
    "STAPHYLOCOCCUS CAPITIS",
    "STAPHYLOCOCCUS WARNERI",
    "STAPHYLOCOCCUS HAEMOLYTICUS",
    "STAPHYLOCOCCUS PETTENKOFERI",
    "STAPHYLOCOCCUS COHNII",
    "STAPHYLOCOCCUS SIMULANS",
    "STAPHYLOCOCCUS SACCHAROLYTICUS",
    "CORYNEBACTERIUM",
    "BACILLUS",
    "CUTIBACTER",
    "PROPIONIBACTER",
    "MICROCOCCUS",
)

TRUE_PATHOGEN_PATTERNS = (
    "STAPH AUREUS",
    "STAPHYLOCOCCUS AUREUS",
    "STAPHYLOCOCCUS LUGDUNENSIS",
    "ENTEROCOCCUS",
    "STREPTOCOCCUS PNEUMONIAE",
    "STREPTOCOCCUS AGALACTIAE",
    "BETA STREPTOCOCCUS GROUP B",
    "STREPTOCOCCUS ANGINOSUS",
    "STREPTOCOCCUS CONSTELLATUS",
    "STREPTOCOCCUS INTERMEDIUS",
    "LISTERIA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA for Gram-positive blood-culture contamination vs true-BSI label validity."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root where reports are written.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="MIMIC-IV root containing hosp/ and icu/. Defaults to MIMIC_IV_ROOT.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for summary JSON. Defaults to <project-root>/reports/blood_culture_label_validity_summary.json.",
    )
    return parser.parse_args()


def _normalize_org(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    if value in {"", "___", "CANCELLED"}:
        return ""
    return value


def _is_gram_positive(org_name: str) -> bool:
    return any(term in org_name for term in GRAM_POSITIVE_TERMS)


def _is_contaminant_like(org_name: str) -> bool:
    if "BACILLUS ANTHRACIS" in org_name:
        return False
    return any(term in org_name for term in CONTAMINANT_PATTERNS)


def _is_true_pathogen_like(org_name: str) -> bool:
    return any(term in org_name for term in TRUE_PATHOGEN_PATTERNS)


def _specimen_category(orgs: list[str]) -> str:
    has_gp = False
    has_contam = False
    has_true = False
    for org in orgs:
        if _is_gram_positive(org):
            has_gp = True
        if _is_contaminant_like(org):
            has_contam = True
        if _is_true_pathogen_like(org):
            has_true = True
    if has_true and has_contam:
        return "mixed_gp"
    if has_true:
        return "true_like"
    if has_contam:
        return "contam_like"
    if has_gp:
        return "ambiguous_gp"
    return "non_gp"


def _read_positive_blood_cultures(config: ProjectConfig) -> pd.DataFrame:
    path = config.hosp_dir / "microbiologyevents.csv"
    frames: list[pd.DataFrame] = []
    usecols = [
        "subject_id",
        "hadm_id",
        "micro_specimen_id",
        "chartdate",
        "charttime",
        "storedate",
        "storetime",
        "spec_type_desc",
        "org_name",
    ]

    reader = pd.read_csv(path, usecols=usecols, chunksize=500_000)
    for chunk in reader:
        chunk = chunk[chunk["spec_type_desc"] == "BLOOD CULTURE"].copy()
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["hadm_id", "micro_specimen_id"]).copy()
        chunk["org_name"] = chunk["org_name"].map(_normalize_org)
        chunk = chunk[chunk["org_name"] != ""].copy()
        if chunk.empty:
            continue
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["storedate"] = pd.to_datetime(chunk["storedate"], errors="coerce")
        chunk["chartdate"] = pd.to_datetime(chunk["chartdate"], errors="coerce")
        chunk["result_time"] = (
            chunk["storetime"]
            .fillna(chunk["charttime"])
            .fillna(chunk["storedate"])
            .fillna(chunk["chartdate"])
        )
        chunk = chunk.dropna(subset=["result_time"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["micro_specimen_id"] = chunk["micro_specimen_id"].astype("int64")
        chunk["subject_id"] = chunk["subject_id"].astype("int64")
        chunk["has_storetime"] = chunk["storetime"].notna().astype(int)
        chunk["has_charttime"] = chunk["charttime"].notna().astype(int)
        frames.append(
            chunk[
                [
                    "subject_id",
                    "hadm_id",
                    "micro_specimen_id",
                    "org_name",
                    "result_time",
                    "has_storetime",
                    "has_charttime",
                ]
            ]
        )

    if not frames:
        raise RuntimeError("No positive blood-culture rows were found.")
    return pd.concat(frames, ignore_index=True)


def _build_specimen_frame(positive_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = positive_rows.sort_values(["hadm_id", "micro_specimen_id", "result_time"]).groupby(
        ["hadm_id", "micro_specimen_id"], as_index=False
    )
    specimen = grouped.agg(
        subject_id=("subject_id", "first"),
        alert_time=("result_time", "min"),
        has_storetime=("has_storetime", "max"),
        has_charttime=("has_charttime", "max"),
        row_count=("org_name", "size"),
        unique_org_count=("org_name", lambda s: int(pd.Series(s).nunique())),
        organisms=("org_name", lambda s: sorted(set(s))),
    )
    specimen["category"] = specimen["organisms"].map(_specimen_category)
    specimen["is_gp_candidate"] = specimen["category"].ne("non_gp").astype(int)
    return specimen.sort_values(["hadm_id", "alert_time", "micro_specimen_id"]).reset_index(drop=True)


def _flag_icu_at_alert(first_alerts: pd.DataFrame, config: ProjectConfig) -> pd.Series:
    icu = pd.read_csv(
        config.icu_dir / "icustays.csv.gz",
        usecols=["hadm_id", "intime", "outtime"],
        compression="gzip",
    )
    icu = icu.dropna(subset=["hadm_id", "intime", "outtime"]).copy()
    icu["hadm_id"] = icu["hadm_id"].astype("int64")
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["intime", "outtime"])

    merged = first_alerts[["hadm_id", "alert_time"]].merge(icu, on="hadm_id", how="left")
    merged["in_icu_at_alert"] = (
        (merged["alert_time"] >= merged["intime"]) & (merged["alert_time"] <= merged["outtime"])
    ).fillna(False)
    flags = merged.groupby(["hadm_id", "alert_time"], as_index=False)["in_icu_at_alert"].max()
    return first_alerts.merge(flags, on=["hadm_id", "alert_time"], how="left")["in_icu_at_alert"].fillna(False)


def _compute_repeat_features(first_alerts: pd.DataFrame, specimen: pd.DataFrame) -> pd.DataFrame:
    specimen_by_hadm: dict[int, list[dict[str, object]]] = {}
    for hadm_id, group in specimen.groupby("hadm_id"):
        specimen_by_hadm[int(hadm_id)] = group[
            ["micro_specimen_id", "alert_time", "organisms", "category"]
        ].to_dict("records")

    repeat_any_48h: list[int] = []
    repeat_gp_48h: list[int] = []
    repeat_same_org_48h: list[int] = []
    repeat_count_48h: list[int] = []

    for row in first_alerts.itertuples(index=False):
        later = []
        for candidate in specimen_by_hadm.get(int(row.hadm_id), []):
            if candidate["micro_specimen_id"] == row.micro_specimen_id:
                continue
            delta = candidate["alert_time"] - row.alert_time
            if pd.Timedelta(0) < delta <= pd.Timedelta(hours=48):
                later.append(candidate)

        first_orgs = set(row.organisms)
        repeat_any = int(bool(later))
        repeat_gp = int(any(candidate["category"] != "non_gp" for candidate in later))
        same_org = int(any(first_orgs.intersection(candidate["organisms"]) for candidate in later))

        repeat_any_48h.append(repeat_any)
        repeat_gp_48h.append(repeat_gp)
        repeat_same_org_48h.append(same_org)
        repeat_count_48h.append(len(later))

    enriched = first_alerts.copy()
    enriched["repeat_any_positive_48h"] = repeat_any_48h
    enriched["repeat_gp_positive_48h"] = repeat_gp_48h
    enriched["repeat_same_organism_48h"] = repeat_same_org_48h
    enriched["repeat_positive_specimen_count_48h"] = repeat_count_48h
    return enriched


def _summarize_first_alerts(first_alerts: pd.DataFrame) -> dict[str, object]:
    category_counts = first_alerts["category"].value_counts().to_dict()
    category_rates = {}
    for category, group in first_alerts.groupby("category"):
        category_rates[str(category)] = {
            "count": int(len(group)),
            "repeat_any_positive_48h": round(100 * group["repeat_any_positive_48h"].mean(), 2),
            "repeat_gp_positive_48h": round(100 * group["repeat_gp_positive_48h"].mean(), 2),
            "repeat_same_organism_48h": round(100 * group["repeat_same_organism_48h"].mean(), 2),
            "in_icu_at_alert": round(100 * group["in_icu_at_alert"].astype(int).mean(), 2),
        }

    high_conf_contam = first_alerts[
        (first_alerts["category"] == "contam_like")
        & (first_alerts["repeat_any_positive_48h"] == 0)
    ].copy()
    high_conf_true = first_alerts[
        (first_alerts["category"] == "true_like")
        & (
            (first_alerts["repeat_any_positive_48h"] == 1)
            | (first_alerts["repeat_same_organism_48h"] == 1)
        )
    ].copy()

    first_org_counts = Counter()
    for orgs in first_alerts["organisms"]:
        for org in orgs:
            if _is_gram_positive(org):
                first_org_counts[org] += 1

    summary = {
        "first_alert_count": int(len(first_alerts)),
        "first_alert_hadm": int(first_alerts["hadm_id"].nunique()),
        "first_alert_with_storetime_pct": round(100 * first_alerts["has_storetime"].mean(), 2),
        "first_alert_in_icu_pct": round(100 * first_alerts["in_icu_at_alert"].astype(int).mean(), 2),
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "category_repeat_rates": category_rates,
        "high_confidence_label_candidates": {
            "likely_contaminant": int(len(high_conf_contam)),
            "likely_true_bsi": int(len(high_conf_true)),
            "indeterminate": int(len(first_alerts) - len(high_conf_contam) - len(high_conf_true)),
        },
        "top_first_alert_organisms": first_org_counts.most_common(15),
    }
    return summary


def build_summary(config: ProjectConfig) -> dict[str, object]:
    positive_rows = _read_positive_blood_cultures(config)
    specimen = _build_specimen_frame(positive_rows)

    gp_specimen = specimen[specimen["is_gp_candidate"] == 1].copy()
    first_alerts = (
        gp_specimen.sort_values(["hadm_id", "alert_time", "micro_specimen_id"])
        .groupby("hadm_id", as_index=False)
        .first()
    )
    first_alerts["in_icu_at_alert"] = _flag_icu_at_alert(first_alerts, config)
    first_alerts = _compute_repeat_features(first_alerts, specimen)

    summary = {
        "cohort_definition": {
            "unit": "First Gram-positive candidate positive blood-culture alert per hospital admission",
            "alert_time": "min(storetime, fallback charttime/storedate/chartdate) at the specimen level",
            "positive_row_definition": "BLOOD CULTURE row with non-empty org_name and org_name != CANCELLED",
        },
        "overall_positive_blood_culture": {
            "positive_rows": int(len(positive_rows)),
            "positive_specimens": int(specimen["micro_specimen_id"].nunique()),
            "positive_hadm": int(specimen["hadm_id"].nunique()),
            "positive_specimens_with_storetime_pct": round(100 * specimen["has_storetime"].mean(), 2),
        },
        "gram_positive_candidate_specimens": {
            "specimens": int(len(gp_specimen)),
            "hadm": int(gp_specimen["hadm_id"].nunique()),
        },
        "first_gram_positive_alerts": _summarize_first_alerts(first_alerts),
    }
    return summary


def main() -> None:
    args = parse_args()
    config = ProjectConfig(project_root=args.project_root, raw_root=args.raw_root)
    json_out = args.json_out or (config.project_root / "reports" / "blood_culture_label_validity_summary.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(config)
    json_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
