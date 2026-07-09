"""
Runs the trained autoencoder (or patchcore, or classifier) over the held-out
test split and spits out a JSON report + a couple of plots. This is what I
point at right before writing up results in the README.

    python -m src.evaluation.evaluate --config configs/config.yaml --mode autoencoder
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset, SyntheticDefectDataset
from src.data.transforms import get_eval_transforms
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import DefectClassifier
from src.models.patchcore import PatchCoreDetector
from src.evaluation.metrics import (
    compute_classification_metrics, find_best_threshold,
    get_confusion_matrix, per_defect_type_breakdown,
)
from src.utils.config import load_config, get_device
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def build_test_set(cfg):
    data_root = Path(cfg.data.root_dir) / cfg.data.category
    eval_tf = get_eval_transforms(cfg.data.image_size, cfg.data.crop_size,
                                   cfg.data.normalize_mean, cfg.data.normalize_std)
    if data_root.exists():
        return MVTecDataset(cfg.data.root_dir, cfg.data.category, "test", eval_tf)
    logger.warning("real dataset not found, evaluating on synthetic data instead")
    return SyntheticDefectDataset(n_samples=150, image_size=cfg.data.crop_size,
                                   transform=eval_tf, defect_ratio=0.4, seed=7)


def _to_unit_range(images, cfg, device):
    mean = torch.tensor(cfg.data.normalize_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.data.normalize_std, device=device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def evaluate_autoencoder(cfg, device):
    ckpt_path = Path(cfg.training.checkpoint_dir) / "best_autoencoder.pt"
    model = ConvAutoencoder(base_channels=cfg.autoencoder.base_channels,
                             latent_dim=cfg.autoencoder.latent_dim).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"loaded checkpoint from epoch {ckpt['epoch']}")
    else:
        logger.warning("no checkpoint found, evaluating an untrained model (numbers will be junk, "
                        "this is just so the pipeline doesn't crash for a fresh clone)")
    model.eval()

    test_set = build_test_set(cfg)
    loader = DataLoader(test_set, batch_size=cfg.data.batch_size, shuffle=False,
                         num_workers=cfg.data.num_workers)

    all_scores, all_labels, all_defect_types = [], [], []
    with torch.no_grad():
        for batch in loader:
            images_01 = _to_unit_range(batch["image"].to(device), cfg, device)
            scores, _ = model.anomaly_map(images_01)
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_defect_types.extend(batch["defect_type"])

    return _finalize_report(all_labels, all_scores, all_defect_types, cfg, tag="autoencoder")


def evaluate_patchcore(cfg, device):
    ckpt_path = Path(cfg.training.checkpoint_dir) / "patchcore_memory_bank.pt"
    detector = PatchCoreDetector(
        backbone=cfg.patchcore.backbone, layers=cfg.patchcore.layers,
        coreset_ratio=cfg.patchcore.coreset_ratio, num_neighbors=cfg.patchcore.num_neighbors,
        device=device,
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found - run training in patchcore mode first")
    detector.load(ckpt_path)

    test_set = build_test_set(cfg)
    loader = DataLoader(test_set, batch_size=cfg.data.batch_size, shuffle=False,
                         num_workers=cfg.data.num_workers)

    all_scores, all_labels, all_defect_types = [], [], []
    for batch in loader:
        scores, _ = detector.score(batch["image"])
        all_scores.extend(scores.numpy().tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_defect_types.extend(batch["defect_type"])

    return _finalize_report(all_labels, all_scores, all_defect_types, cfg, tag="patchcore")


def evaluate_classifier(cfg, device):
    ckpt_path = Path(cfg.training.checkpoint_dir) / "best_model.pt"
    model = DefectClassifier(
        backbone=cfg.model.backbone, pretrained=False, num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_set = build_test_set(cfg)
    loader = DataLoader(test_set, batch_size=cfg.data.batch_size, shuffle=False,
                         num_workers=cfg.data.num_workers)

    all_scores, all_labels, all_defect_types = [], [], []
    with torch.no_grad():
        for batch in loader:
            probs = model.predict_proba(batch["image"].to(device))
            defect_prob = probs[:, 1]
            all_scores.extend(defect_prob.cpu().numpy().tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_defect_types.extend(batch["defect_type"])

    return _finalize_report(all_labels, all_scores, all_defect_types, cfg, tag="classifier")


def _finalize_report(labels, scores, defect_types, cfg, tag):
    labels = np.array(labels)
    scores = np.array(scores)

    # normalize scores to 0-1 so the report threshold is easy to reason about
    if scores.max() > scores.min():
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = scores

    best_thresh = find_best_threshold(labels, scores_norm) if len(set(labels)) > 1 else 0.5
    metrics = compute_classification_metrics(labels, scores_norm, threshold=best_thresh)
    cm = get_confusion_matrix(labels, scores_norm, threshold=best_thresh)
    breakdown = per_defect_type_breakdown(defect_types, labels, scores_norm, threshold=best_thresh)

    report_dir = Path(cfg.evaluation.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model": tag,
        "category": cfg.data.category,
        "n_test_samples": len(labels),
        "best_threshold": float(best_thresh),
        "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else round(v, 4))
                    for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "per_defect_type": breakdown,
    }

    with open(report_dir / f"{tag}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    _plot_score_distribution(labels, scores_norm, report_dir / f"{tag}_score_dist.png")

    logger.info(f"[{tag}] metrics: {report['metrics']}")
    logger.info(f"report written to {report_dir / f'{tag}_report.json'}")
    return report


def _plot_score_distribution(labels, scores, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(scores[labels == 0], bins=30, alpha=0.6, label="good")
    plt.hist(scores[labels == 1], bins=30, alpha=0.6, label="defective")
    plt.xlabel("normalized anomaly score")
    plt.ylabel("count")
    plt.title("anomaly score distribution: good vs defective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["classifier", "autoencoder", "patchcore"],
                         default="autoencoder")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.project.device)

    if args.mode == "autoencoder":
        evaluate_autoencoder(cfg, device)
    elif args.mode == "patchcore":
        evaluate_patchcore(cfg, device)
    elif args.mode == "classifier":
        evaluate_classifier(cfg, device)


if __name__ == "__main__":
    main()
