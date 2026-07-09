"""
Generic-ish trainer for the supervised classifier. Kept the autoencoder
training loop separate (see train_autoencoder in train.py) since the loss
and metrics are different enough that mashing them into one class made the
code harder to follow, not easier.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.utils.logger import get_logger

logger = get_logger("trainer")


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=cfg.training.log_dir)

    def _build_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.cfg.training.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.cfg.training.lr,
                                      weight_decay=self.cfg.training.weight_decay)
        # falling back to plain adam if someone changes the config, don't want a hard crash
        return torch.optim.Adam(params, lr=self.cfg.training.lr)

    def _build_scheduler(self):
        if self.cfg.training.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.training.epochs
            )
        return None

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()

        total_loss, correct, n = 0.0, 0, 0
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                if train:
                    self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.cfg.training.mixed_precision):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                n += images.size(0)

        return total_loss / n, correct / n

    def fit(self):
        logger.info(f"training on {self.device}, {len(self.train_loader.dataset)} train / "
                     f"{len(self.val_loader.dataset)} val samples")

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"epoch {epoch:03d}/{self.cfg.training.epochs} | "
                f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val_loss {val_loss:.4f} acc {val_acc:.3f} | {elapsed:.1f}s"
            )

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("acc/train", train_acc, epoch)
            self.writer.add_scalar("acc/val", val_acc, epoch)

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, val_loss)
            else:
                self.patience_counter += 1

            if epoch % self.cfg.training.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch}.pt", epoch, val_loss)

            if self.patience_counter >= self.cfg.training.early_stopping_patience:
                logger.info(f"no improvement for {self.patience_counter} epochs, stopping early")
                break

        self.writer.close()
        logger.info(f"done. best val loss = {self.best_val_loss:.4f}")

    def _save_checkpoint(self, filename, epoch, val_loss):
        path = Path(self.cfg.training.checkpoint_dir) / filename
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "val_loss": val_loss,
            "config": dict(self.cfg),
        }, path)
