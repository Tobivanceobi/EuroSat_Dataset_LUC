import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.nn.functional import interpolate


def overlay_cam_on_image(img, cam_mask):
    """
    Overlays the CAM mask on the image as a heatmap.

    Args:
    - img (Tensor): The original image tensor of shape [C, H, W].
    - cam_mask (Tensor): The CAM mask tensor of shape [H, W].

    Returns:
    - combined_img (Tensor): The image with the CAM overlay.
    """
    # Normalize the CAM mask to be in [0, 1]
    cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min())

    # Resize the CAM mask to match the image size
    cam_mask = interpolate(cam_mask.unsqueeze(0), size=img.shape[1:], mode='bilinear',
                           align_corners=False).squeeze(0)

    # Convert CAM mask to heatmap
    heatmap = plt.get_cmap('jet')(cam_mask.cpu().detach().numpy())[:, :, :3]  # Get the RGB part, discard alpha
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float()

    # Overlay the heatmap on the image
    combined_img = heatmap * 0.3 + img.cpu() * 0.5  # Adjust opacity as needed

    return combined_img



