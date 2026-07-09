"""
Entry point for training. Run as:

    python -m src.training.train --config configs/config.yaml --mode classifier
    python -m src.training.train --config configs/config.yaml --mode autoencoder
    python -m src.training.train --config configs/config.yaml --mode patchcore

Defaults to synthetic data if data/mvtec/<category> isn't found on disk, so
this is runnable straight out of the box for anyone checking out the repo -
grab the real dataset (see scripts/download_mvtec.py) once you actually want
real numbers.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import MVTecDataset, SyntheticDefectDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.classifier import DefectClassifier
from src.models.autoencoder import ConvAutoencoder
from src.models.patchcore import PatchCoreDetector
from src.training.trainer import Trainer
from src.utils.config import load_config, set_seed, get_device
from src.utils.logger import get_logger

logger = get_logger("train")


def build_datasets(cfg):
    data_root = Path(cfg.data.root_dir) / cfg.data.category
    train_tf = get_train_transforms(cfg.data.image_size, cfg.data.crop_size,
                                     cfg.data.normalize_mean, cfg.data.normalize_std)
    eval_tf = get_eval_transforms(cfg.data.image_size, cfg.data.crop_size,
                                   cfg.data.normalize_mean, cfg.data.normalize_std)

    if data_root.exists():
        logger.info(f"found real dataset at {data_root}")
        full_train = MVTecDataset(cfg.data.root_dir, cfg.data.category, "train", train_tf)
        test_set = MVTecDataset(cfg.data.root_dir, cfg.data.category, "test", eval_tf)
    else:
        logger.warning(f"{data_root} not found, falling back to synthetic data - "
                        f"see scripts/download_mvtec.py to get the real dataset")
        full_train = SyntheticDefectDataset(n_samples=400, image_size=cfg.data.crop_size,
                                             transform=train_tf, defect_ratio=0.0)  # train = good only
        test_set = SyntheticDefectDataset(n_samples=120, image_size=cfg.data.crop_size,
                                           transform=eval_tf, defect_ratio=0.4, seed=99)

    val_len = int(len(full_train) * cfg.data.val_split)
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])

    return train_set, val_set, test_set


def train_classifier(cfg, device):
    train_set, val_set, _ = build_datasets(cfg)

    # note: for the pure classifier path we actually need defective examples
    # in the training data too (unlike autoencoder/patchcore). if you're using
    # the synthetic fallback for this mode, bump defect_ratio in build_datasets.
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True,
                               num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.data.batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers)

    model = DefectClassifier(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        freeze_layers=cfg.model.freeze_backbone_layers,
    )

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.fit()


def train_autoencoder(cfg, device):
    train_set, val_set, _ = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True,
                               num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.data.batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers)

    model = ConvAutoencoder(base_channels=cfg.autoencoder.base_channels,
                             latent_dim=cfg.autoencoder.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                   weight_decay=cfg.training.weight_decay)
    criterion = torch.nn.MSELoss()

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            # note: autoencoder wants inputs in [0,1], not imagenet-normalized,
            # since we're reconstructing pixels directly - see denorm below
            images_01 = _to_unit_range(images, cfg)

            optimizer.zero_grad()
            recon = model(images_01)
            loss = criterion(recon, images_01)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images_01 = _to_unit_range(batch["image"].to(device), cfg)
                recon = model(images_01)
                val_loss += criterion(recon, images_01).item() * images_01.size(0)
        val_loss /= len(val_loader.dataset)

        logger.info(f"epoch {epoch:03d}/{cfg.training.epochs} | "
                     f"train_mse {train_loss:.5f} | val_mse {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       checkpoint_dir / "best_autoencoder.pt")
        else:
            patience += 1

        if patience >= cfg.training.early_stopping_patience:
            logger.info("early stopping triggered")
            break

    logger.info(f"done. best val mse = {best_val:.5f}")


def _to_unit_range(images, cfg):
    # reverses the imagenet normalization the dataloader applies, since the
    # autoencoder is happier reconstructing plain [0,1] pixel values
    mean = torch.tensor(cfg.data.normalize_mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.data.normalize_std, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def fit_patchcore(cfg, device):
    train_set, _, _ = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=False,
                               num_workers=cfg.data.num_workers)

    detector = PatchCoreDetector(
        backbone=cfg.patchcore.backbone,
        layers=cfg.patchcore.layers,
        coreset_ratio=cfg.patchcore.coreset_ratio,
        num_neighbors=cfg.patchcore.num_neighbors,
        device=device,
    )
    logger.info("fitting patchcore memory bank on good training samples...")
    detector.fit(train_loader)
    logger.info(f"memory bank size: {detector.memory_bank.shape[0]} patch embeddings")

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    detector.save(checkpoint_dir / "patchcore_memory_bank.pt")
    logger.info(f"saved to {checkpoint_dir / 'patchcore_memory_bank.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["classifier", "autoencoder", "patchcore"],
                         default="classifier")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    if args.mode == "classifier":
        train_classifier(cfg, device)
    elif args.mode == "autoencoder":
        train_autoencoder(cfg, device)
    elif args.mode == "patchcore":
        fit_patchcore(cfg, device)


if __name__ == "__main__":
    main()
