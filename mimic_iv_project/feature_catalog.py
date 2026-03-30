from __future__ import annotations

import json
import re
from dataclasses import asdict

import pandas as pd

from .config import FEATURE_SPECS, ProjectConfig


def _read_dictionary(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")


def build_catalog(config: ProjectConfig) -> dict:
    d_items = _read_dictionary(config.icu_dir / "d_items.csv.gz")
    d_labitems = _read_dictionary(config.hosp_dir / "d_labitems.csv")

    catalog = {
        "config": {
            "history_hours": config.history_hours,
            "future_hours": config.future_hours,
            "bin_hours": config.bin_hours,
        },
        "features": [],
    }

    for spec in FEATURE_SPECS:
        dictionary = d_labitems if spec.source == "lab" else d_items
        pattern = "|".join(spec.regexes)
        matched = dictionary[dictionary["label"].fillna("").str.contains(pattern, case=False, regex=True)].copy()
        matched = matched.drop_duplicates(subset=["itemid"]).sort_values("itemid")

        feature_entry = asdict(spec)
        feature_entry["itemids"] = matched["itemid"].astype(int).tolist()
        feature_entry["matched_labels"] = matched[["itemid", "label"]].to_dict(orient="records")
        catalog["features"].append(feature_entry)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(config.catalog_path, "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)

    return catalog


def load_catalog(config: ProjectConfig) -> dict:
    with open(config.catalog_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def feature_names_by_source(catalog: dict, source: str) -> list[str]:
    return [feature["name"] for feature in catalog["features"] if feature["source"] == source]

