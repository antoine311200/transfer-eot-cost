import torch


def is_image_tensor(x: torch.Tensor) -> bool:
    """Check if tensor is image-shaped (4D: batch, channels, height, width)."""
    return x.dim() == 4
