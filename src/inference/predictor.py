"""
Wraps up loading + preprocessing + inference into one object so both the
FastAPI service and the streamlit demo can just do:

    predictor = DefectPredictor(cfg)
    result = predictor.predict(pil_image)

instead of duplicating this logic in two places (learned that lesson after
having to fix the same bug twice in two different files on an earlier project).

supports all three model types now - autoencoder, patchcore, classifier.
patchcore is the one that actually performed well on real data (0.994 auroc
vs the autoencoder's 0.566, see README results section) so that's the one
worth defaulting the demo to.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import DefectClassifier
from src.models.patchcore import PatchCoreDetector
from src.utils.gradcam import GradCAM
from src.utils.visualization import overlay_heatmap


class DefectPredictor:
    def __init__(self, cfg, mode="patchcore", device=None):
        self.cfg = cfg
        self.mode = mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_eval_transforms(
            cfg.data.image_size, cfg.data.crop_size,
            cfg.data.normalize_mean, cfg.data.normalize_std,
        )
        self._load_model()

    def _load_model(self):
        ckpt_dir = Path(self.cfg.training.checkpoint_dir)

        if self.mode == "autoencoder":
            self.model = ConvAutoencoder(
                base_channels=self.cfg.autoencoder.base_channels,
                latent_dim=self.cfg.autoencoder.latent_dim,
            ).to(self.device)
            ckpt_path = ckpt_dir / "best_autoencoder.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()

        elif self.mode == "patchcore":
            self.model = PatchCoreDetector(
                backbone=self.cfg.patchcore.backbone,
                layers=self.cfg.patchcore.layers,
                coreset_ratio=self.cfg.patchcore.coreset_ratio,
                num_neighbors=self.cfg.patchcore.num_neighbors,
                device=self.device,
            )
            ckpt_path = ckpt_dir / "patchcore_memory_bank.pt"
            if ckpt_path.exists():
                self.model.load(ckpt_path)
            # note: if there's no memory bank on disk, .score() will raise -
            # that's intentional, patchcore genuinely has nothing to compare
            # against without having fit() on some good training images first

        elif self.mode == "classifier":
            self.model = DefectClassifier(
                backbone=self.cfg.model.backbone, pretrained=False,
                num_classes=self.cfg.model.num_classes, dropout=self.cfg.model.dropout,
            ).to(self.device)
            ckpt_path = ckpt_dir / "best_model.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.gradcam = GradCAM(self.model)

        else:
            raise ValueError(f"predictor doesn't support mode={self.mode} yet")

    def _to_unit_range(self, tensor):
        mean = torch.tensor(self.cfg.data.normalize_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.cfg.data.normalize_std, device=self.device).view(1, 3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)

    def predict(self, image: Image.Image, return_heatmap=True):
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        if self.mode == "autoencoder":
            with torch.no_grad():
                unit_tensor = self._to_unit_range(tensor)
                score, error_map = self.model.anomaly_map(unit_tensor)
            score = float(score.item())
            heatmap = error_map.squeeze().cpu().numpy()
            # normalize for display, raw reconstruction error isn't 0-1 bounded
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # heads up - this threshold is just a guess, not calibrated against
            # the actual score distribution. the raw score number is what
            # actually means something right now, the good/defective flag is
            # more of a rough gut check until I wire up real calibration
            is_defective = score > getattr(self, "_ae_threshold", 0.01)

        elif self.mode == "patchcore":
            # patchcore wants the same imagenet-normalized tensor the
            # classifier uses (it's just running a pretrained backbone over
            # it), no unit-range conversion needed like the autoencoder
            image_scores, heatmaps = self.model.score(tensor)
            score = float(image_scores.item())
            heatmap = heatmaps.squeeze().numpy()
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # same deal as the autoencoder threshold above - this is a rough
            # guess, not pulled from the actual calibrated threshold in
            # reports/patchcore_report.json (that one's computed on scores
            # normalized across the whole test set, which isn't directly
            # comparable to a single raw score at inference time - would need
            # to persist the test set's raw score min/max to fix this properly)
            is_defective = score > getattr(self, "_patchcore_threshold", 15.0)

        elif self.mode == "classifier":
            probs = self.model.predict_proba(tensor)[0]
            score = float(probs[1].item())
            is_defective = score >= self.cfg.evaluation.threshold
            heatmap, _ = self.gradcam.generate(tensor) if return_heatmap else (None, None)

        result = {
            "is_defective": bool(is_defective),
            "score": score,
            "mode": self.mode,
        }

        if return_heatmap and heatmap is not None:
            overlay = overlay_heatmap(image, heatmap)
            result["heatmap_overlay"] = overlay   # PIL image, caller decides how to serialize it

        return result