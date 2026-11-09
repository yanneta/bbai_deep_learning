# HW4 – Part 2: Building a Personalized Model with Implicit Feedback

In Part 2 of the assignment, you will build on the work completed in [Part 1](part1.md) to create a more advanced personalized ranking recommender system. While Part 1 focused on simple baseline models (popularity and item–item similarity), Part 2 introduces negative sampling, learned embeddings, and fitting a personalized model.

## Learning Objectives

By the end of Part 2, you will be able to:

- Use negative sampling for implicit feedback datasets.
- Train a Matrix Factorization model.
- Generate personalized top-K recommendations.
- Compare your trained model to the Part 1 baselines.
- Evaluate performance.

## Required Inputs (from Part 1)

You must reuse your saved artifacts from Part 1:

- `train.csv`
- `test.csv`
- Evaluation function
- Baseline results for comparison

## Negative Sampling

Implicit-feedback datasets contain only positive interactions (e.g., clicks, views, purchases). To train a model, we also need negative examples—items the user did not interact with.

In this assignment, you will generate negative samples using a simple uniform sampling strategy:

Let `all_items` be the set of all item IDs in the dataset.

For each positive interaction `(u, pos_item)` in `train.csv`:

- Randomly sample an item `neg_item` from `all_items`. (For simplicity, do not worry if this item appears in the user's history—the probability is extremely small.)

For each positive interaction, create two training rows:

- `(u, pos_item, label=1)` — the real interaction
- `(u, neg_item, label=0)` — the sampled negative example

### Expected Output

After negative sampling, your training data should contain one row for each positive interaction and one row for its sampled negative item.

Use the following format:

```
user_id | item_id | label
```

Where:

- `label = 1` for real (positive) interactions
- `label = 0` for sampled negative items

## Preprocessing

Reindex users and items for the use of embeddings.

## Model

You will train a simple matrix factorization model using learned user and item embeddings.

```python
u_vec = embedding_user[user_id]
i_vec = embedding_item[item_id]
```

The predicted score for a user–item pair is the dot product:

```
pred(u, i) = u_vec · i_vec
```

This score represents how likely the user is to interact with the item.

## Training the Model

You will train the model using the `(user_id, item_id, label)` dataset created above.

- Use binary cross-entropy loss.
- Train for several epochs. Go over the full dataset multiple times to allow embeddings to converge.
- Track the training loss. The loss should generally decrease across epochs. If it doesn't, adjust:
  1. learning rate,
  2. number of epochs,
  3. embedding dimension.

This code gets the numpy arrays into tensors:

```python
user_tensor = torch.from_numpy(train_user).long()
item_tensor = torch.from_numpy(train_item).long()
label_tensor = torch.from_numpy(train_label).float()
```

## 4. Generating Recommendations

After training, for each user:

1. Compute the predicted score for all candidate items.
2. Exclude items already seen in training.
3. Return the top-K highest scoring items.

You should implement a function with the signature:

```python
def recommend_model(user_id, k):
    """Return top-k item_ids recommended by the trained model."""
```

**Note:** In this homework you will compute recommendations by scoring all items and sorting them. In real production systems (Spotify, YouTube, TikTok, Amazon), this step is performed using Approximate Nearest Neighbor (ANN) search libraries such as FAISS or ScaNN. These libraries find the highest-scoring items in the embedding space without scoring every item, enabling millisecond-level recommendations even with millions of items.

## 5. Evaluation

Use the same metrics as in Part 1.

Compare:

- Your trained model
- Popularity baseline
- Item–item similarity baseline

Visualize results in a small table or bar chart.

## Deliverables

Jupyter notebook with:

- Negative sampling implementation
- Model training code
- Recommendation generation
- Evaluation
- Comparison with baselines
