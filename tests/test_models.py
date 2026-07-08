"""
Shape/sanity checks on the model forward passes. Not testing accuracy here
(that needs real data + training time), just making sure tensors flow
through without blowing up and come out the right shape.
"""

import torch

from src.models.classifier import DefectClassifier
from src.models.autoencoder import ConvAutoencoder


def test_classifier_output_shape():
    model = DefectClassifier(backbone="resnet18", pretrained=False, num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 2)


def test_classifier_predict_proba_sums_to_one():
    model = DefectClassifier(backbone="resnet18", pretrained=False, num_classes=2)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    probs = model.predict_proba(x)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


def test_classifier_frozen_layers_have_no_grad():
    model = DefectClassifier(backbone="resnet18", pretrained=False, freeze_layers=2)
    frozen_params = list(model.stem.parameters()) + list(model.layer1.parameters())
    assert all(not p.requires_grad for p in frozen_params)
    # later layers should still be trainable
    assert any(p.requires_grad for p in model.layer4.parameters())


def test_autoencoder_reconstructs_same_shape():
    model = ConvAutoencoder(base_channels=16, latent_dim=64)
    x = torch.rand(2, 3, 224, 224)   # autoencoder expects [0,1] range, not imagenet-normalized
    recon = model(x)
    assert recon.shape == x.shape


def test_autoencoder_anomaly_map_shapes():
    model = ConvAutoencoder(base_channels=16, latent_dim=64)
    x = torch.rand(3, 3, 224, 224)
    score, heatmap = model.anomaly_map(x)
    assert score.shape == (3,)
    assert heatmap.shape == (3, 1, 224, 224)


def test_autoencoder_output_in_valid_range():
    model = ConvAutoencoder(base_channels=16, latent_dim=64)
    x = torch.rand(1, 3, 224, 224)
    recon = model(x)
    assert recon.min() >= 0.0 and recon.max() <= 1.0
