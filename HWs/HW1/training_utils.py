import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from image_manipulation import apply_brightness, apply_contrast, rgb_to_grayscale,  \
        horizontal_flip, center_crop, _crop_by_coords, random_crop, add_black_border, \
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
    image = image.reshape(-1, 1, 28, 28)
    N = image.shape[0]
    if level == "mild":
        image = random_crop(image, 26, 26)
        if torch.rand(1).item() < flip_prob:
            image = horizontal_flip(image)
    elif level == "moderate":
        image = random_crop(image, 24, 24)
        if torch.rand(1).item() < flip_prob:
            image = horizontal_flip(image)
        image = apply_contrast(image, 0.01)
    elif level == "aggressive":
        image = random_crop(image, 20, 20)
        if torch.rand(1).item() < flip_prob:
            image = horizontal_flip(image)
        image = apply_brightness(image, 0.1)
        image = apply_contrast(image, 0.1) 
    return image.reshape(N, -1)


def apply_augmentations_val(image, level="mild"):
    """
    Apply center_crop to validation to meet the same crop level as training.

    Args:
        image (torch.Tensor): Input image tensor (C, H, W)
        level (str): One of "mild", "moderate", or "aggressive"

    Returns:
        torch.Tensor: Augmented image
    """
    image = image.reshape(-1, 1, 28, 28)
    N = image.shape[0]
    if level == "mild":
        image = center_crop(image, 26, 26)
    elif level == "moderate":
        image = center_crop(image, 24, 24)
    elif level == "aggressive":
        image = center_crop(image, 20, 20)
    return image.reshape(N, -1)


def show_batch(images):
    N = images.shape[0]
    plt.figure(figsize=(8, 8))
    for i in range(N):
        img = images[i]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
