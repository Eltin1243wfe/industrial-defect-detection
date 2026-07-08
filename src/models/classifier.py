"""
Supervised path: ResNet backbone + custom head for good/defective classification.

This is the "if you actually have labeled defective examples" model. In most
real factory settings you don't have many of those (defects are rare by
definition) which is why I also built the unsupervised autoencoder and
patchcore paths in this same package - see autoencoder.py and patchcore.py.
Use whichever fits the amount of labeled data you've actually got.
"""

import torch
import torch.nn as nn
from torchvision import models


_BACKBONES = {
    "resnet18": (models.resnet18, 512),
    "resnet34": (models.resnet34, 512),
    "resnet50": (models.resnet50, 2048),
}


class DefectClassifier(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, num_classes=2,
                 dropout=0.3, freeze_layers=0):
        super().__init__()

        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone}, pick from {list(_BACKBONES)}")

        ctor, feat_dim = _BACKBONES[backbone]
        weights = "IMAGENET1K_V1" if pretrained else None
        net = ctor(weights=weights)

        # chop off the avgpool+fc, I want the raw feature maps for grad-cam later
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self._maybe_freeze(freeze_layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def _maybe_freeze(self, n):
        # freezing early layers - they're just learning edges/textures anyway,
        # no point re-training those on a small defect dataset
        layers = [self.stem, self.layer1, self.layer2, self.layer3]
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feats = self.layer4(x)   # keeping this around for grad-cam hooks
        return feats

    def forward(self, x):
        feats = self.forward_features(x)
        pooled = self.avgpool(feats)
        logits = self.head(pooled)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
