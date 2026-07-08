"""
Unsupervised anomaly detection via reconstruction error.

Idea: train only on "good" images. A well-trained autoencoder learns to
reconstruct normal texture cleanly, but chokes on stuff it's never seen
(scratches, dents, contamination) so the reconstruction error spikes there.
That gives us both an image-level anomaly score (mean error) and a rough
pixel-level localization map (per-pixel error) for free, no defect labels
needed at training time at all.

This is the model I'd actually reach for first on a new product line, since
labeled defect data is almost always scarce early on.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_dim=256):
        super().__init__()

        c = base_channels
        self.encoder = nn.Sequential(
            self._down_block(in_channels, c),          # 224 -> 112
            self._down_block(c, c * 2),                 # 112 -> 56
            self._down_block(c * 2, c * 4),              # 56 -> 28
            self._down_block(c * 4, c * 8),              # 28 -> 14
        )
        self.bottleneck_in = nn.Conv2d(c * 8, latent_dim, kernel_size=1)
        self.bottleneck_out = nn.Conv2d(latent_dim, c * 8, kernel_size=1)

        self.decoder = nn.Sequential(
            self._up_block(c * 8, c * 4),                # 14 -> 28
            self._up_block(c * 4, c * 2),                 # 28 -> 56
            self._up_block(c * 2, c),                     # 56 -> 112
            self._up_block(c, c),                          # 112 -> 224
        )
        self.out_conv = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()   # input is normalized to [0,1] before feeding in, see note in trainer

    @staticmethod
    def _down_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _up_block(in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck_in(z)
        z = self.bottleneck_out(z)
        recon = self.decoder(z)
        recon = self.out_act(self.out_conv(recon))
        return recon

    @torch.no_grad()
    def anomaly_map(self, x):
        """
        per-pixel reconstruction error, averaged over channels.
        returns (anomaly_score, heatmap) where score is a single float per image
        and heatmap is the same H,W as input (useful for overlay visualization).
        """
        recon = self.forward(x)
        error = (x - recon).pow(2).mean(dim=1, keepdim=True)   # B,1,H,W
        score = error.mean(dim=(1, 2, 3))                       # B,
        return score, error
