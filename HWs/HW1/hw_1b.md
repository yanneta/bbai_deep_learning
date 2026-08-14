## Hw 1c: Experimental Analysis of a Shallow Neural Network

Neural networks are powerful function approximators, but their performance depends heavily on the choice of architecture and training parameters. In this part of the assignment, you will systematically explore how different hyperparameters and training strategies impact the performance of a shallow neural network trained on the Fashion-MNIST dataset. These experiments will help you develop an intuition for model tuning and performance tradeoffs in real-world settings.

We will use the **Adam** optimizer for all experiments. The dataset contains **grayscale 32×32 images** of clothing items, and your task is to classify each image into one of **10 categories**. You are encouraged to reuse and extend your code from **Part 1b**.

---

### 1. Pick a Validation Set

Use the **first 10,000 examples** from the training set as your **validation set**. The remaining examples will be used for training.

---

### 2. Learning Rate Selection

Train a 2-layer neural network with the following configuration:

- Hidden layer size: `50`
- Activation: ReLU
- Optimizer: Adam
- Epochs: 500

Train separate models using the following learning rates:

`[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]`


For each setting, **reinitialize the model from scratch** (i.e., do not reuse weights).

Then, **interpolate between the best two values** by testing additional learning rates (e.g., 0.005, 0.002, etc.) to identify the optimal learning rate.

Train each model for 500 epochs and report the training loss, validation loss, and validation accuracy as a function of learning rate in a table.


---

### 3. Effect of Hidden Layer Size

Fix the learning rate to the best value found above. Now investigate how the size of the hidden layer affects model performance.

Train models with the following hidden sizes:

`[10, 50, 100, 300, 1000, 2000]`

Train each model for 500 epochs and report the training loss, validation loss, and validation accuracy as a function of hidden size in a table.

---
### 4. Visualize transformed images

Pick a set of transformations (e.g., random crop, flip, moderate brightness/contrast) and visualize four images along with their transformed versions.

---

### 5. Effect of Data Augmentation

In this section, you will explore how **data augmentation** affects model performance using the functions you implemented earlier.

Use the **best learning rate** and **best hidden size** from your previous experiments.

Design a set of **augmentation pipelines**, each applying a different combination and intensity of augmentations. For example:

- **Baseline**: no augmentations
- **Mild**: horizontal flip, slight brightness/contrast adjustments
- **Moderate**: random crop, flip, moderate brightness/contrast
- **Aggressive**: all of the above + black borders and stronger intensity

You may use or modify the `apply_augmentations` function from `training_utils.py`. Use `apply_augmentations_val` to ensure that validation images are cropped to the same size as training images (if applicable).

Train a model for each augmentation level and report the training loss, validation loss, and validation accuracy as a function of transformation level in a table.

---

## Deliverables

- Use `training_utils.py` to define any reusable code, such as:
  - A function that trains a model and logs results
  - A function that returns a freshly initialized model
  - Any augmentation functions

- A **notebook** that runs all experiments and presents tables and plots. The notebook should include **minimal inline code**: most logic should be imported from `training_utils.py`.



### Advice: Why You Should Clone the Data Before Augmentation
In PyTorch, tensors are mutable. This means that if you apply an augmentation (like cropping or flipping) directly to your training data, you might accidentally modify the original data — permanently. This can lead to unexpected bugs, especially if you try to reuse the same data in future epochs or experiments.

For example:
```
# This modifies X_train in-place if apply_augmentations does not clone internally
X_train = apply_augmentations(X_train, level="moderate")
```

Later, if you try to reshape or augment X_train again assuming it's 28×28, it might already be cropped to 24×24 or 20×20 — causing shape errors like:

```
RuntimeError: shape '[-1, 1, 28, 28]' is invalid for input of size ...
```

To avoid this, always create a copy of the original training data before applying any augmentations:
```
X_train_aug = apply_augmentations(X_train.clone(), level=level)
```
