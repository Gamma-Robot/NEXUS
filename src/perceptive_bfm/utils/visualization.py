import os
import sys

import cv2
import torch


def debug_visualize_image(
    image: torch.Tensor,
    scale_up_vis: int = 5,
    window_name: str = "vis_image",
) -> None:
    """Visualize images in a cv2 window for debugging purposes.

    This function normalizes images to [0, 255], handles different channel configurations,
    scales them up for better visualization, and displays them in an OpenCV window.

    Args:
        images: Image tensor in shape (H, W).
        scale_up_vis: The factor to scale up the image for better visualization if the
            resolution is too low. Defaults to 5.
        window_name: The name of the OpenCV window. Defaults to "vis_image".
    """
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return

    # automatically normalize images to [0, 255]
    img = (image * 255.0 / image.max() + 1e-8).cpu().numpy().astype("uint8")  # (H, W)
    # Scale up the image for better visualization
    img = cv2.resize(
        img,
        (img.shape[1] * scale_up_vis, img.shape[0] * scale_up_vis),
        interpolation=cv2.INTER_AREA,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)
