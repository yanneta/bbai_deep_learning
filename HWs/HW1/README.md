# Deep Learning Homework 1: Image Classification with PyTorch

This repository contains the first homework assignment for the Deep Learning course. In this assignment, you'll walk through the full deep learning pipeline for image classification using PyTorch—from implementing custom image augmentations, to training a shallow neural network, and finally experimenting with architectural and training variations to analyze their impact on performance.

---

## Part 1a: Batched Image Manipulation with PyTorch Tensors

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

### Part 1b: Training a Linear Classifier on Fashion-MNIST
In this part, you'll build and train a simple image classifier using PyTorch on the Fashion-MNIST dataset, which contains grayscale images of clothing items. You'll implement a linear model (or shallow neural network) to classify images into 10 categories, while practicing model definition, training loops, evaluation, and performance measurement.


### Deliverables
Implement all required code in `image_classification_shallow.ipynb`. 


### Part 1c: Experimental Analysis of a Shallow Neural Network

Neural networks are flexible function approximators, but their performance is highly sensitive to the choice of architecture and training parameters. In this part of the assignment, you will systematically study how different hyperparameters and training strategies affect the performance of a shallow neural network trained on the Fashion-MNIST dataset. Your experiments will help you understand practical model tuning and performance tradeoffs in real-world settings.

Find the instructions for the assignment [here](hw_1c.md)

