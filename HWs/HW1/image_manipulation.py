"""
Deep Learning Course - Assignment: Image Manipulation with PyTorch Tensors

Student Name: _______________
Date: _______________

INSTRUCTIONS:
This assignment focuses on manipulating images using PyTorch tensor operations.
You will implement various image filters and transformations using only tensor
operations (no external image processing libraries like PIL filters, OpenCV, etc.).

Image Format:
- Images are represented as tensors with shape (C, H, W) where:
  - C = channels (3 for RGB, 1 for grayscale)  
  - H = height in pixels
  - W = width in pixels
- Pixel values are normalized to range [0, 1]

Rules:
1. Use only PyTorch tensor operations
2. Do not use external image processing libraries for the core operations
3. Ensure your functions work with batched inputs when specified
4. Handle edge cases appropriately
5. All functions should preserve the input tensor's device and dtype

"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def apply_brightness(image: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    """
    Adjust the brightness of an image.
    
    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W)
        brightness_factor: Factor to adjust brightness. 
                          1.0 = no change, >1.0 = brighter, <1.0 = darker
    
    Returns:
        Brightness-adjusted image tensor with same shape as input
        
    TODO: Implement brightness adjustment by scaling pixel values.
          Remember to clamp values to [0, 1] range.
    """
    pass  # Your code here


def apply_contrast(image: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    """
    Adjust the contrast of an image.
    
    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W)
        contrast_factor: Factor to adjust contrast.
                        1.0 = no change, >1.0 = more contrast, <1.0 = less contrast
    
    Returns:
        Contrast-adjusted image tensor with same shape as input
        
    TODO: Implement contrast adjustment using the formula:
          new_pixel = (pixel - 0.5) * contrast_factor + 0.5
          Remember to clamp values to [0, 1] range.
    """
    pass  # Your code here


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to grayscale.
    
    Args:
        image: RGB input tensor of shape (3, H, W) or (B, 3, H, W)
    
    Returns:
        Grayscale image tensor of shape (1, H, W) or (B, 1, H, W)
        
    TODO: Convert RGB to grayscale using the standard weights:
          gray = 0.299 * R + 0.587 * G + 0.114 * B
    """
    pass  # Your code here


def horizontal_flip(image: torch.Tensor) -> torch.Tensor:
    """
    Flip image horizontally (mirror effect).
    
    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        Horizontally flipped image tensor with same shape as input
        
    TODO: Flip the image along the width dimension.
          Hint: Use tensor indexing with [::-1] or torch.flip()
    """
    pass  # Your code here


def _crop_by_coords(image: torch.Tensor, top: int, left: int, crop_height: int, crop_width: int) -> torch.Tensor:
    """
    Crop a region starting at (top, left) with size (crop_height, crop_width).
    Supports (C, H, W) or (B, C, H, W).
    """
    pass  # Your code here


def center_crop(image: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    """
    Crop the center region of an image.

    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W)
        crop_height: Height of the crop region
        crop_width: Width of the crop region

    Returns:
        Cropped image tensor of shape (C, crop_height, crop_width) or
        (B, C, crop_height, crop_width)

    TODO: Extract the center region of the specified size.
          Handle cases where crop size is larger than image size.
    """
    pass  # Your code here


def random_crop(image: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    """
    Perform a random crop of the specified size on the input image tensor.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W).
        crop_height (int): Height of the crop region.
        crop_width (int): Width of the crop region.

    Returns:
        torch.Tensor: Cropped image tensor of shape (C, crop_height, crop_width) or
                      (N, C, crop_height, crop_width), matching the input format.

    Notes:
        - The crop location is chosen randomly for each image in the batch.
        - If a single image is passed, a single random crop is returned.
        - This function uses PyTorch's random number generator; set the seed via
          `torch.manual_seed` for reproducible results.
    """
    pass  # Your code here


def add_black_border(image, border_size):
    """
    Adds a black border around the input image tensor.

    Parameters:
    -----------
    image : torch.Tensor
        A 3D (C, H, W) or 4D (N, C, H, W) tensor representing an image or batch of images.
        - C must be 3 (for RGB).
    border_size : int
        The number of pixels to add to each side of the image.

    Returns:
    --------
    torch.Tensor
        A new tensor with black borders added. The output shape will be:
        - (3, H + 2*border_size, W + 2*border_size) for single images
        - (N, 3, H + 2*border_size, W + 2*border_size) for batches

    Instructions for implementing:
    ------------------------------
    1. Extract the height (H) and width (W) from the input image shape.
    2. Create a new tensor filled with zeros (black pixels), with the expanded size.
    3. Copy the original image into the center of the new tensor using slicing.
       For example, if `border_size` is 2, then the original image goes from
       index 2 to 2+H along the height axis.
    4. Support both 3D (single image) and 4D (batch) input.

    """
    pass  # Your code here


def numpy_to_torch_batch(image_list):
    """
    Converts a list of NumPy images (H, W, C) to a PyTorch batch tensor (N, C, H, W).

    Parameters:
    -----------
    image_list : list of np.ndarray
        List of NumPy arrays representing images, each of shape (H, W, C)

    Returns:
    --------
    torch.Tensor
        A 4D tensor of shape (N, C, H, W), dtype float32 and values scaled to [0, 1] if original was uint8
    """
    pass  # Your code here
