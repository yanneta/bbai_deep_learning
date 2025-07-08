import torch

from image_manipulation import apply_brightness, apply_contrast, rgb_to_grayscale,  \
        horizontal_flip, crop_center, _crop_by_coords, random_crop, add_black_border, \
        numpy_to_torch_batch

def apply_augmentations(image, level="mild", flip_prob=0.5):
    """
    Apply a sequence of data augmentations based on the specified intensity level.

    Args:
        image (torch.Tensor): Input image tensor (C, H, W)
        level (str): One of "mild", "moderate", or "aggressive"
        flip_prob (float): Probability of applying horizontal flip

    Returns:
        torch.Tensor: Augmented image
    """
    if level == "mild":
        if torch.rand(1).item() < flip_prob:
            image = horizontal_flip(image)
        image = apply_brightness(image, factor=0.1)
    
    elif level == "moderate":
        image = random_crop(image, 30, 30)
        if torch.rand(1).item() < flip_prob:
            image = horizontal_flip(image)
        image = apply_brightness(image, factor=0.2)
        image = apply_contrast(image, factor=0.2)
    
    elif level == "aggressive":
        image = random_crop(image, 24, 24)
        if torch.rand(1).item() < flip_prob:
            image = horizontal_flip(image)
        image = add_black_border(image, thickness=4)
        image = apply_brightness(image, factor=0.3)
        image = apply_contrast(image, factor=0.3) 
    return image


def apply_augmentations_val(image, level="mild"):
    """
    Apply center_crop to validation to meet the same crop level as training.

    Args:
        image (torch.Tensor): Input image tensor (C, H, W)
        level (str): One of "mild", "moderate", or "aggressive"

    Returns:
        torch.Tensor: Augmented image
    """
    if level == "moderate":
        image = center_crop(image, 30, 30)
    elif level == "aggressive":
        image = center_crop(image, 24, 24)
    return image
