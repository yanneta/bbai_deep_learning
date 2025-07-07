# 🧠 Deep Learning Homework 1: Image Manipulation with PyTorch

This repository contains the first homework assignment for the Deep Learning course. The assignment is split into three parts, each focused on using **PyTorch** for low-level tensor manipulation, with an emphasis on working directly with raw image data.

---

## ✅ Part 1a: Batched Image Manipulation with PyTorch Tensors

### Deliverables
Implement all required functions in: `image_manipulation.py`

### Overview

In this part, you'll implement common image transformations **from scratch** using **PyTorch tensor operations**. You will **not** use any high-level image processing libraries like PIL, OpenCV, or torchvision.

### Image Format

Your functions must handle inputs in the following formats:

- Single image: `(C, H, W)`
- Batch of images: `(N, C, H, W)`

Where:
- `N`: number of images in the batch  
- `C`: number of channels (1 for grayscale, 3 for RGB)  
- `H`, `W`: height and width of the image  
- Pixel values are normalized in `[0, 1]`

### Requirements

- All functions must accept both **batched** and **unbatched** inputs.
- Use only **PyTorch tensor operations** (no PIL, OpenCV, or torchvision).
- Preserve the input tensor's **`dtype`**.
- Use `tensor.ndim` or `tensor.dim()` to distinguish input shapes.

### Testing

We’ve provided a test script:

```bash
pytest test_image_manipulation.py
```

