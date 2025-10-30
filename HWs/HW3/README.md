# Homework 3: Convolutional Neural Networks

## Part 3a: Tiny CNN on MNIST

**Goal:** Build and train a *very small* CNN in PyTorch on a *tiny* MNIST subset. Keep it simple and fast (works on CPU).

### Files
- `tiny_cnn.ipynb`

### Instructions
1. Open the starter notebook.
2. Run the setup/data cells to download MNIST automatically.
3. Complete the **SmallCNN** and the **training loop** sections where marked.
4. Train for **5 epochs**. Aim for **>90%** validation accuracy on this tiny split (it’s typically achievable).
5. Fill out the **Mini-Experiments** answers in Markdown.
6. Submit the executed notebook (with outputs visible).

### Expected Runtime
On CPU this should complete in just a few minutes. On GPU it’s faster.

### Grading Rubric (10 pts)
- (3 pts) Correct small CNN; shapes pass; param count shown.
- (3 pts) Training loop works without errors; reasonable learning curve.
- (2 pts) Test evaluation + confusion matrix.
- (2 pts) Mini-Experiments answered with short justifications.

---

## Part 3b: Transfer Learning on Oxford Flowers 102

In this lab, you will use a **pretrained MobileNetV2** model and fine-tune it to classify flower species from the **Oxford 102 Flowers** dataset.

---

## Dataset

Use the built-in dataset:

```python
from torchvision.datasets import Flowers102

* Use at least 600 images for training and 300 for testing.
* Here is a link that might be useful: https://dev.to/hyperkai/flowers102-in-pytorch-4l71
* Replace the final classifier layer so the model outputs 102 classes.
* Achieve around 90% validation accuracy.
* Show 5 example predictions (some correct and some incorrect).
* Experiment with augmentations not seen in class. (in my notebook)
