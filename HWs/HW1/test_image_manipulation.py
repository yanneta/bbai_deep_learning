from image_manipulation import apply_brightness, apply_contrast, rgb_to_grayscale,  \
        horizontal_flip, crop_center, _crop_by_coords, random_crop, add_black_border, \
        numpy_to_torch_batch
import torch
from torch.testing import assert_close
import numpy as np

def create_test_image() -> torch.Tensor:
    """Create a simple test image for testing functions."""
    # Create a simple 3x32x32 RGB test image with some patterns
    image = torch.zeros(3, 32, 32)
    
    # Add some colored regions
    image[0, 8:24, 8:24] = 1.0  # Red square
    image[1, 12:20, 12:20] = 1.0  # Green square (overlapping)
    image[2, 16:32, 0:16] = 1.0  # Blue region
    
    # Add some gradient
    for i in range(32):
        image[:, i, :] += i / 32.0 * 0.3
    
    return torch.clamp(image, 0, 1)


def test_brightness():
    """Test brightness adjustment function."""
    image = create_test_image()
    
    # Test brightening
    bright = apply_brightness(image, 1.5)
    assert bright.shape == image.shape, "Shape should be preserved"
    assert torch.all(bright >= image), "Brightened image should have higher values"
    assert torch.all(bright <= 1.0), "Values should be clamped to [0, 1]"
    
    # Test darkening
    dark = apply_brightness(image, 0.5)
    assert torch.all(dark <= image), "Darkened image should have lower values"
    assert torch.all(dark >= 0.0), "Values should be clamped to [0, 1]"
    

def test_contrast():
    """Test contrast adjustment function."""
    image = create_test_image()
    
    high_contrast = apply_contrast(image, 2.0)
    assert high_contrast.shape == image.shape, "Shape should be preserved"
    assert torch.all(high_contrast >= 0.0) and torch.all(high_contrast <= 1.0), "Values should be in [0, 1]"
    
    low_contrast = apply_contrast(image, 0.5)
    assert torch.all(low_contrast >= 0.0) and torch.all(low_contrast <= 1.0), "Values should be in [0, 1]"
    

def test_grayscale():
    """Test RGB to grayscale conversion."""
    image = create_test_image()  # 3x32x32
    
    gray = rgb_to_grayscale(image)
    assert gray.shape == (1, 32, 32), f"Expected (1, 32, 32), got {gray.shape}"
    assert torch.all(gray >= 0.0) and torch.all(gray <= 1.0), "Values should be in [0, 1]"
    

def test_horizontal_flip():
    """Test horizontal flip function."""
    print("Testing horizontal_flip...")
    image = create_test_image()
    
    flipped = horizontal_flip(image)
    assert flipped.shape == image.shape, "Shape should be preserved"
    
    # Test that flipping twice returns original
    double_flipped = horizontal_flip(flipped)
    assert torch.allclose(image, double_flipped, atol=1e-6), "Double flip should return original"
    

def test_crop_center():
    """Test center cropping function."""
    print("Testing crop_center...")
    image = create_test_image()  # 3x32x32
    
    cropped = crop_center(image, 16, 16)
    assert cropped.shape == (3, 16, 16), f"Expected (3, 16, 16), got {cropped.shape}"

def test_crop_by_coords_3d_basic():
    img = torch.arange(3 * 5 * 5).reshape(3, 5, 5)
    crop = _crop_by_coords(img, top=1, left=1, crop_height=3, crop_width=3)
    expected = img[:, 1:4, 1:4]
    assert crop.shape == (3, 3, 3)
    assert torch.equal(crop, expected)

def test_crop_by_coords_4d_basic():
    batch = torch.arange(2 * 3 * 5 * 5).reshape(2, 3, 5, 5)
    crop = _crop_by_coords(batch, top=2, left=0, crop_height=2, crop_width=3)
    expected = batch[:, :, 2:4, 0:3]
    assert crop.shape == (2, 3, 2, 3)
    assert torch.equal(crop, expected)

def test_crop_by_coords_edge_crop():
    img = torch.arange(3 * 4 * 4).reshape(3, 4, 4)
    crop = _crop_by_coords(img, top=3, left=2, crop_height=1, crop_width=2)
    expected = img[:, 3:4, 2:4]
    assert crop.shape == (3, 1, 2)
    assert torch.equal(crop, expected)

def test_add_black_border():
    # Test case 1: Single 3x3 RGB image with border=1
    img = torch.ones(3, 3, 3)
    result = add_black_border(img, border_size=1)
    assert result.shape == (3, 5, 5)
    # Check that the center matches the original image
    assert_close(result[:, 1:4, 1:4], img)
    # Check that the border is all zeros
    assert (result[:, 0, :] == 0).all()
    assert (result[:, -1, :] == 0).all()
    assert (result[:, :, 0] == 0).all()
    assert (result[:, :, -1] == 0).all()

    # Test case 2: Batch of 2 RGB images, size 4x4, border=2
    img_batch = torch.ones(2, 3, 4, 4)
    result = add_black_border(img_batch, border_size=2)
    assert result.shape == (2, 3, 8, 8)
    assert_close(result[:, :, 2:6, 2:6], img_batch)
    assert (result[:, :, 0:2, :] == 0).all()
    assert (result[:, :, -2:, :] == 0).all()
    assert (result[:, :, :, 0:2] == 0).all()
    assert (result[:, :, :, -2:] == 0).all()

    # Test case 3: No border
    img = torch.rand(3, 8, 8)
    result = add_black_border(img, border_size=0)
    assert result.shape == (3, 8, 8)
    assert_close(result, img)

    # Test case 4: Custom dtype and device
    if torch.cuda.is_available():
        img = torch.ones(3, 4, 4, device='cuda', dtype=torch.float32)
        result = add_black_border(img, border_size=1)
        assert result.device == torch.device('cuda')
        assert result.dtype == torch.float32

def test_random_crop_output_shape_3d():
    img = torch.randn(3, 64, 64)  # C, H, W
    crop = random_crop(img, crop_height=32, crop_width=32)
    assert crop.shape == (3, 32, 32)

def test_random_crop_output_shape_4d():
    batch = torch.randn(4, 3, 64, 64)  # N, C, H, W
    crop = random_crop(batch, crop_height=32, crop_width=32)
    assert crop.shape == (4, 3, 32, 32)

def test_numpy_to_torch_batch():
    images = [np.random.rand(16, 16, 3).astype(np.float32) for _ in range(2)]
    batch = numpy_to_torch_batch(images)
    assert batch.shape == (2, 3, 16, 16)
    assert batch.dtype == torch.float32
    assert batch.max() <= 1.0 and batch.min() >= 0.0

