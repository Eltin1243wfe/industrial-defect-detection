"""
Simplified version of PatchCore (Roth et al., 2022) - "Towards Total Recall in
Industrial Anomaly Detection". This is the approach I've seen get the best
results in practice for surface defect detection, so wanted it in here even
though it's the most involved of the three model paths.

High level:
  1. run good training images through a frozen, pretrained CNN
  2. grab mid-level feature maps (layer2/layer3 - early enough to keep spatial
     detail, late enough to be semantically meaningful)
  3. treat every spatial location's feature vector as a "patch embedding"
  4. build a memory bank of all those patch embeddings, subsampled via
     greedy coreset selection so it doesn't blow up in size
  5. at inference: embed the test image the same way, and for every patch
     find its nearest neighbor distance to the memory bank. Far away =
     never seen a normal patch like this = probably a defect.

No backprop needed at "training" time here, which is nice - fitting the
memory bank on a few hundred images takes well under a minute on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PatchCoreExtractor(nn.Module):
    """wraps a pretrained backbone and returns concatenated mid-level feature maps"""

    def __init__(self, backbone="wide_resnet50_2", layers=("layer2", "layer3")):
        super().__init__()
        net = getattr(models, backbone)(weights="IMAGENET1K_V2")
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        self.layers = layers
        self._features = {}
        for name in layers:
            getattr(net, name).register_forward_hook(self._make_hook(name))

        self.net = net

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._features[name] = out
        return hook

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        self.net(x)   # we don't care about final logits, just the hooked intermediate maps

        # resize everything to the size of the first requested layer, then stack channel-wise
        ref_size = self._features[self.layers[0]].shape[-2:]
        maps = []
        for name in self.layers:
            fmap = self._features[name]
            if fmap.shape[-2:] != ref_size:
                fmap = F.interpolate(fmap, size=ref_size, mode="bilinear", align_corners=False)
            maps.append(fmap)

        combined = torch.cat(maps, dim=1)   # B, C_total, H, W
        return combined


def greedy_coreset_subsample(embeddings: torch.Tensor, ratio: float = 0.1, n_start=10):
    """
    Greedy coreset selection - iteratively picks the embedding that's farthest
    from the current subset, so the final memory bank stays diverse instead of
    just being a bunch of near-duplicate patches from similar-looking images.

    embeddings: (N, D) tensor of all patch embeddings pooled across the train set
    returns: (M, D) subsampled tensor, M ~= ratio * N
    """
    n = embeddings.shape[0]
    target = max(int(n * ratio), n_start)
    target = min(target, n)

    device = embeddings.device
    selected_idx = torch.randint(0, n, (1,), device=device)
    selected = embeddings[selected_idx]

    # running min-distance from every point to the selected set so far
    min_dist = torch.cdist(embeddings, selected).squeeze(1)

    picked = [selected_idx.item()]
    for _ in range(target - 1):
        next_idx = torch.argmax(min_dist)
        picked.append(next_idx.item())
        new_dist = torch.cdist(embeddings, embeddings[next_idx].unsqueeze(0)).squeeze(1)
        min_dist = torch.minimum(min_dist, new_dist)

    return embeddings[picked]


class PatchCoreDetector:
    """
    Not an nn.Module on purpose - there's no learned weights beyond the
    frozen backbone, the "model" is really just the memory bank + a distance
    lookup at inference. Keeping it as a plain class made this way less
    confusing to reason about when I was debugging it.
    """

    def __init__(self, backbone="wide_resnet50_2", layers=("layer2", "layer3"),
                 coreset_ratio=0.1, num_neighbors=9, device="cpu"):
        self.extractor = PatchCoreExtractor(backbone, layers).to(device).eval()
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.device = device
        self.memory_bank = None   # populated by fit()
        self.feature_map_size = None

    @torch.no_grad()
    def _embed_batch(self, images):
        fmap = self.extractor(images.to(self.device))   # B, C, H, W
        b, c, h, w = fmap.shape
        self.feature_map_size = (h, w)
        # reshape to (B*H*W, C) - every spatial location becomes its own "patch"
        patches = fmap.permute(0, 2, 3, 1).reshape(-1, c)
        return patches

    def fit(self, dataloader, max_patches_before_coreset=20000):
        all_patches = []
        for batch in dataloader:
            patches = self._embed_batch(batch["image"])
            all_patches.append(patches.cpu())
        all_patches = torch.cat(all_patches, dim=0)
        print(f"pulled {all_patches.shape[0]} patch embeddings out of the train set")

        # this is the part that was killing my runtime - greedy coreset is
        # O(N) per pick and I was running it on the full patch pool (160k+
        # patches for something as small as bottle). random-subsampling down
        # to a manageable pool FIRST, then running greedy selection on that,
        # cuts the wait from ~an hour down to like a minute or two and barely
        # changes the memory bank quality since it's still a decent spread
        # of the training distribution
        if all_patches.shape[0] > max_patches_before_coreset:
            keep_idx = torch.randperm(all_patches.shape[0])[:max_patches_before_coreset]
            all_patches = all_patches[keep_idx]
            print(f"randomly capped down to {all_patches.shape[0]} before coreset "
                  f"(this is the step that used to take forever)")

        self.memory_bank = greedy_coreset_subsample(all_patches, self.coreset_ratio).to(self.device)
        return self

    @torch.no_grad()
    def score(self, images):
        """returns (image_scores, pixel_heatmaps) for a batch of images"""
        if self.memory_bank is None:
            raise RuntimeError("call .fit() on good training data before scoring anything")

        b = images.shape[0]
        patches = self._embed_batch(images)   # (B*H*W, C)
        dists = torch.cdist(patches, self.memory_bank)   # (B*H*W, M)
        nn_dists, _ = dists.topk(self.num_neighbors, dim=1, largest=False)
        patch_scores = nn_dists.mean(dim=1)   # average of k nearest neighbor distances

        h, w = self.feature_map_size
        heatmaps = patch_scores.reshape(b, h, w)
        image_scores = heatmaps.reshape(b, -1).max(dim=1).values   # max patch anomaly = image score

        return image_scores.cpu(), heatmaps.cpu()

    def save(self, path):
        torch.save({
            "memory_bank": self.memory_bank.cpu(),
            "feature_map_size": self.feature_map_size,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.memory_bank = ckpt["memory_bank"].to(self.device)
        self.feature_map_size = ckpt["feature_map_size"]
        return self
