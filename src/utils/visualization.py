"""
Small plotting helpers - mainly for overlaying anomaly heatmaps (from
autoencoder / patchcore / gradcam, doesn't matter which, they're all just
2D arrays by the time they get here) on top of the original image.
"""

import numpy as np
import matplotlib as mpl
from PIL import Image


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha=0.45, colormap="jet"):
    """
    image: PIL RGB image
    heatmap: 2D numpy array, values in [0,1], any resolution (gets resized to match image)
    returns: PIL RGB image with heatmap overlaid
    """
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heatmap_np = np.array(heatmap_img) / 255.0

    # cm.get_cmap() got axed in newer matplotlib versions, this is the way
    # that actually still works across versions - grabbing straight from
    # the colormaps registry instead
    cmap = mpl.colormaps[colormap]
    colored = cmap(heatmap_np)[:, :, :3]   # drop alpha channel from cmap output
    colored = (colored * 255).astype(np.uint8)

    base = np.array(image).astype(np.float32)
    blended = base * (1 - alpha) + colored.astype(np.float32) * alpha
    return Image.fromarray(blended.astype(np.uint8))

def side_by_side(image: Image.Image, overlay: Image.Image) -> Image.Image:
    """slap the original and the overlaid heatmap next to each other, handy for reports"""
    w, h = image.size
    canvas = Image.new("RGB", (w * 2 + 10, h), color=(255, 255, 255))
    canvas.paste(image, (0, 0))
    canvas.paste(overlay, (w + 10, 0))
    return canvas
