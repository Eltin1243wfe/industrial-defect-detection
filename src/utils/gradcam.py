"""
Grad-CAM (Selvaraju et al.) hooked onto the classifier's last conv block.
Wanted this mainly so I'm not just outputting "defective: 87%" with nothing
to back it up - being able to show *where* the model thinks the defect is
makes the demo way more convincing, and it's genuinely useful for debugging
when the model is right for the wrong reasons (background texture instead
of the actual defect, etc).

Only wired up for DefectClassifier right now since it needs a proper conv
backbone with gradients flowing - the autoencoder/patchcore paths get their
localization for free from the reconstruction error / patch distance maps
instead, see anomaly_map() and PatchCoreDetector.score().
"""

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        # layer4 is the last conv block before pooling, standard choice for grad-cam
        self.target_layer = model.layer4
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image_tensor, target_class=None):
        """
        image_tensor: (1, C, H, W), already normalized, requires no grad set beforehand
        returns: heatmap resized to the input's H,W, values in [0,1]
        """
        image_tensor = image_tensor.clone().requires_grad_(True)

        logits = self.model(image_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # global-average-pool the gradients to get per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)   # only care about features that positively influence the target class

        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # normalize to 0-1 for easy overlay later
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class
