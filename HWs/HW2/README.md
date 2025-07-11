# Homework 2: Sliding Window Neural Network for Named Entity Recognition (NER)

Named Entity Recognition (NER) is a core task in natural language processing, where the goal is to identify spans of text that refer to entities such as people, locations, or proteins. In this assignment, you'll implement a **sliding window neural network** for NER.

Your model will take a window of 5 consecutive words and predict the NER tag of the **middle word** in the window. This approach allows the model to incorporate both left and right context when making a prediction.

---

## Part 1: Implementation

Follow the steps below to build your NER pipeline. Each step corresponds to a function or module you'll implement.

### 1. Encode the Dataset

Implement a function `build_vocab_mapping`, `build_label_mapping`, `encode_dataset()` that:
- `build_vocab_mapping` computes `word_to_ix` dictionary
- `build_label_mapping` returns `label_to_ix`

`encode_dataset()`:
- Converts tokens to indices using a vocabulary
- Converts NER labels to integers

```python

def build_vocab_mapping(words):


def build_label_mapping(labels):


def encode_dataset(words, labels, word_to_ix, label_to_ix):
```

Note: the `word_to_ix` and `label_to_ix` are built with the training data. Use the last 50000 lines as validation and the rest as training.

### 2 Create Sliding Window Dataset

Implement a WindowDataset class that:

- Takes encoded token sequences and labels

- Produces input windows of 5 consecutive words (as word indices)

- Assigns the label of the middle word as the target

```python
class WindowDataset(Dataset):
    def __init__(self, encoded_words, encoded_labels, window_size=5):
        super().__init__()
        ...
    def __getitem__(self, idx):
        return window, label
```
Hint: make window a `np.array`.

### 3. Build the Model

Implement a simple feedforward neural network with the following structure:

- Embedding Layer: First, map each word index to a dense vector using `nn.Embedding`. The output will be of shape (batch_size, window_size, embedding_dim).

- Concatenation: Flatten the embeddings into a single vector of shape (batch_size, window_size * embedding_dim).

- Hidden Layer: A linear layer followed by a ReLU activation and optional dropout.

- Output Layer: A final linear layer that outputs scores for each NER class.

This architecture allows the model to learn a contextual representation of the center word using the surrounding words in the window.

```python
class WindowNERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, window_size, num_classes):
        super().__init__()
        ...
    def forward(self, x):
        ...

```

### 4. Write F1-macro scores

Write a function `compute_f1_per_class` that computes the F1-score for each class given true and predicted labels.

For every class compute:

F1-score formula:
F1 = 2 · (Precision · Recall) / (Precision + Recall)

To compute it:

Precision for class k = correctly predicted k / total predicted as k

Recall for class k = correctly predicted k / total ground truth k



### 5. Training and Evaluation
Split the data by using the last 50000 lines as validation and the rest as training.



Write training and validation functions:

```python
def train_model(model, dataloader, optimizer, epochs):
    ...

def valid_metrics(model, dataloader):
    ...

```

You should track **Training loss**,**Validation loss** and  **Validation accuracy**.

# Part 2: Experimentation

Now that your model works, explore how different settings affect performance.



**Baseline Configuration**:
Unless otherwise specified, use the following as your baseline:
- Epocs: 10 
- Learning rate: 0.001
- Embedding dimension: 50
- Hidden layer size: 30
- Dropout: 0.0
- Window size: 5


For all experiments produce a table that shows training loss, validation loss, validation accuracy and validation F1-macro.
Here is a code that you can use an example:

```python
print("| Emb Dimension | Training Loss | Validation Loss | Validation Acc | F1-macro |")
print("|---------------|---------------|-----------------|----------------|----------|")

for emb_dim, metric in zip(embedding_dims, metrics):
    loss, val_loss, acc, f1 = metric
    print(f"| {emb_dim:<13} | {loss:<13.4f} | {val_loss:<15.4f} | {acc:<15.2f} | {f1:<15.2f} |")
```

### Experiment 1: Embedding Dimension
Train models with different embedding sizes (e.g., 50, 100, 200) and compare validation accuracy.

### Experiment 2: Hidden Layer Size
Try hidden layer sizes like 10, 30, 100, 300 and evaluate their effect.

### Experiment 3: Dropout
Add dropout after the hidden layer and test values like 0.0, 0.2, 0.5.

### Experiment 4: Window Size
Try a larger window (e.g., 7 or 9 words) and see how it affects accuracy.

## Reporting
For each experiment, include:

- A short table of hyperparameters and validation accuracy

- One plot showing validation accuracy over epochs

A short reflection: What worked best? Why?

##  Deliverables
`ner_utils.py`: functions and classes (encoding, dataset, model, training)

`ner_experiments.ipynb`: your notebook with experiments, results, and commentary

(Optional) plots/ folder with charts of accuracy/loss


