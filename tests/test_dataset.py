"""
Using the synthetic dataset for tests since we can't count on the real
MVTec data being present in CI. Just want to catch obvious breakage here,
not validate model quality.
"""

import torch

from src.data.dataset import SyntheticDefectDataset
from src.data.transforms import get_train_transforms, get_eval_transforms


def test_synthetic_dataset_length():
    ds = SyntheticDefectDataset(n_samples=50, image_size=64)
    assert len(ds) == 50


def test_synthetic_dataset_item_shapes():
    tf = get_eval_transforms(image_size=72, crop_size=64)
    ds = SyntheticDefectDataset(n_samples=10, image_size=64, transform=tf)
    item = ds[0]

    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape[0] == 1
    assert item["label"].item() in (0, 1)


def test_defect_ratio_roughly_respected():
    ds = SyntheticDefectDataset(n_samples=500, defect_ratio=0.3, seed=1)
    defective_frac = sum(ds.labels) / len(ds.labels)
    # not exact since it's randomly sampled, just checking it's in the right ballpark
    assert 0.2 < defective_frac < 0.4


def test_train_transforms_produce_tensor():
    tf = get_train_transforms(image_size=80, crop_size=64)
    ds = SyntheticDefectDataset(n_samples=5, image_size=64, transform=tf)
    item = ds[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 64, 64)


def test_mask_is_nonzero_for_defective_samples():
    ds = SyntheticDefectDataset(n_samples=100, defect_ratio=1.0, seed=3)   # force all defective
    item = ds[0]
    assert item["label"].item() == 1
    assert item["mask"].sum() > 0
