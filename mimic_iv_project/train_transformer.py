from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ProjectConfig, TASK_COLUMNS
from .metrics import binary_auprc, binary_auroc, binary_brier
from .models import PatientSpecificDecayTransformer


class SequenceDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.values = torch.from_numpy(arrays["values"][indices]).float()
        self.masks = torch.from_numpy(arrays["masks"][indices]).float()
        self.counts = torch.from_numpy(arrays["counts"][indices]).float()
        self.deltas = torch.from_numpy(arrays["deltas"][indices]).float()
        self.static_features = torch.from_numpy(arrays["static_features"][indices]).float()
        self.labels = torch.from_numpy(arrays["labels"][indices]).float()

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int):
        return (
            self.values[index],
            self.masks[index],
            self.counts[index],
            self.deltas[index],
            self.static_features[index],
            self.labels[index],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the patient-specific decay transformer.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _split_indices(cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        split: cohort.index[cohort["split"] == split].to_numpy(dtype=np.int64)
        for split in ["train", "val", "test"]
    }


def _task_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auroc": binary_auroc(y_true, y_prob),
        "auprc": binary_auprc(y_true, y_prob),
        "brier": binary_brier(y_true, y_prob),
    }


def evaluate(model, loader, device) -> tuple[float, dict]:
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for values, masks, counts, deltas, static_features, labels in loader:
            logits = model(
                values.to(device),
                masks.to(device),
                counts.to(device),
                deltas.to(device),
                static_features.to(device),
            )
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    metrics = {}
    score_values = []
    for task_index, task_name in enumerate(TASK_COLUMNS):
        task_metric = _task_metrics(labels[:, task_index], probs[:, task_index])
        metrics[task_name] = task_metric
        if not np.isnan(task_metric["auroc"]):
            score_values.append(task_metric["auroc"])
    mean_score = float(np.mean(score_values)) if score_values else float("nan")
    return mean_score, metrics


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else ProjectConfig().project_root
    config = ProjectConfig(project_root=project_root)

    data = np.load(config.sequence_dataset_path)
    arrays = {key: data[key] for key in data.files}
    cohort = pd.read_csv(config.cohort_path)
    with open(config.sequence_metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    split_indices = _split_indices(cohort)
    train_dataset = SequenceDataset(arrays, split_indices["train"])
    val_dataset = SequenceDataset(arrays, split_indices["val"])
    test_dataset = SequenceDataset(arrays, split_indices["test"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = PatientSpecificDecayTransformer(
        num_dynamic_features=arrays["values"].shape[-1],
        num_static_features=arrays["static_features"].shape[-1],
        num_tasks=len(metadata["task_names"]),
        num_time_bins=arrays["values"].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_score = float("-inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for values, masks, counts, deltas, static_features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(
                values.to(device),
                masks.to(device),
                counts.to(device),
                deltas.to(device),
                static_features.to(device),
            )
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * values.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        val_score, val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_auroc": val_score})

        if val_score > best_score:
            best_score = val_score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_mean_auroc={val_score:.4f}"
        )
        print(json.dumps(val_metrics, indent=2))

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_metrics = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), config.artifacts_dir / "patient_decay_transformer.pt")
    with open(config.artifacts_dir / "transformer_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(config.artifacts_dir / "transformer_test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
