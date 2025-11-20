# HW4: Recommender System with Implicit Feedback


# Part 1 — Building a Baseline Recommender System

This assignment introduces you to recommender systems built on **implicit feedback**—data that reflects *user behavior* (clicks, views, likes) rather than explicit ratings. You will work with a subset of the **Tenrec (Tencent)** dataset and build simple, interpretable baseline recommendation models.

---

## Learning Objectives
By the end of Part 1, you will be able to:

- Understand the difference between explicit and implicit feedback.
- Clean and preprocess real-world interaction logs.
- Build a **user–item interaction history** suitable for recommender systems.
- Construct a **train/test split** appropriate for sequential implicit data.
- Implement two baseline models:
  - **Popularity-based recommender**
  - **Item–item similarity recommender**
- Evaluate recommendations using **top-K metrics**.
- Interpret results and compare baseline performance.

---

## Dataset
We will use task 2 from the **Tenrec** dataset (`task_2.csv` file).

You should extract the following fields after applying filtering (see below):
- `user_id`
- `item_id`
- `timestamp`
---

## 1. Data Preprocessing
Your first goal is to build a clean interaction log.

### Steps:
1. Load the raw interaction file for the selected Tenrec task.
2. Remove malformed rows or missing IDs.
3. Filter:
   - Users with fewer than **5 interactions**
   - Items with fewer than **5 interactions**
3. Create timestamp by `df["timestamp"] = df.groupby("user_id").cumcount()`
4. Sort interactions by timestamp.

Advise: Use `value_counts()` + boolean mask to filter. I first filtered by user and the by items. I get 209197 interactions.


Expected output: a cleaned DataFrame with the structure:
```
user_id,item_id,timestamp
```

You will reuse this cleaned dataset in **Part 2**, so save it as a file.

---

## 2. Train/Test Split (Leave-One-Out)
For each user:
- Use **all but the last** interaction as the training set.
- Use the **last interaction** as the test set.

```
df = df.sort_values(["user_id", "timestamp"])
test = df.groupby("user_id").tail(1)
test_idx = test.index
train = df.loc[~df.index.isin(test_idx)]
```

This ensures you predict what the user interacted with *next*, simulating real-world behavior.

Expected outputs:
- `train.csv`
- `test.csv`

---

## 3. Baseline Models
You will implement two simple but important baseline recommenders.

### **A. Popularity-Based Recommender**
Recommends the items with the highest overall number of interactions.

Steps:
1. Count occurrences of each `item_id` in `train`.
2. Recommend the top-K most frequent items **not already seen** by the user.

```
def already_seen(train_df, user_id):
    """ Returns a list of items already seen by the user
    """


def recommend_popularity(train_df, user_id, k):
    """
    Recommend the top-k most popular items the user has not interacted with.

    Inputs
    ----------
    train_df : pandas.DataFrame
    user_id: User for whom we want recommendations.
    k : Number of items to recommend.

    Returns
    -------
    list: A list of item_ids of length k (or fewer if not enough items exist).
    """
```


### **B. Item–Item Similarity Recommender**
Recommends items that commonly co-occur with the user's history.

Approach (choose one):
- **Co-occurrence counts** from user histories

Steps:
1. Build dictionary with keys are `item_id` and values are the list of users that interacted with that item.
`item_users = train.groupby("item_id")["user_id"].apply(set).to_dict()`
2. Compute cooc(i, j) = number of users who interacted with BOTH i and j.
3. Recommend the top-K items excluding those already seen.

---

## 4. Evaluation Metrics
Implement **top-K recommendation metrics**, ideally for K ∈ {5, 10, 20}.

Required metric:
- **Recall@K** — did the model recommend the next item.

Compare both baseline models using these metrics.

```
def recall_at_k(recommended_items, true_item, k):
    """
    Compute Recall@K for a single user.

    Parameters
    ----------
    recommended_items : list
        A list of item_ids recommended to the user (length >= k).
    true_item : int or str
        The actual next item the user interacted with (from the test set).
    k : int
        The cutoff for evaluation (e.g., 5, 10, 20).

    Returns
    -------
    float
        1.0 if the true_item is in the top-k recommendations, otherwise 0.0.
    """
```
---

## 5. What You Must Submit
Your Part 1 submission includes:

### Code (Jupyter Notebook)
- Preprocessing steps
- Train/test construction
- Baseline models
- Evaluation functions
- Performance comparison plot or table

---
# Part 2 -- Building a Personalized Model with Implicit Feedback

In Part 2 of the assignment, you will build on the work completed in **Part 1** to create a more advanced **personalized ranking recommender system**. While Part 1 focused on simple baseline models (popularity and item–item similarity), Part 2 introduces **negative sampling**, **learned embeddings**, and fitting a personalized model.

---

## Learning Objectives
By the end of Part 2, you will be able to:

- Use negative sampling for implicit feedback datasets.
- Train a Matrix Factorization Model model:
- Generate personalized top-K recommendations.
- Compare your trained model to the Part 1 baselines.
- Evaluate performance.
---

## Required Inputs (from Part 1)
You must reuse your saved artifacts from Part 1:

- `train.csv`
- `test.csv`
- Evaluation function
- Baseline results for comparison

---
##  Negative Sampling

Implicit-feedback datasets contain only positive interactions (e.g., clicks, views, purchases).
To train a model, we also need negative examples—items the user did not interact with.

In this assignment, you will generate negative samples using a simple uniform sampling strategy:

Let `all_items` be the set of all item IDs in the dataset.

For each positive interaction `(u, pos_item)` in train.csv:

Randomly sample an item `neg_item` from `all_items`.
(For simplicity, do not worry if this item appears in the user’s history—the probability is extremely small.)

For each positive interaction, create two training rows:

`(u, pos_item, label=1)` — the real interaction

`(u, neg_item, label=0)` — the sampled negative example

## Expected Output

After negative sampling, your training data should contain one row for each positive interaction and one row for its sampled negative item.

Use the following format:

`user_id | item_id | label`

Where:
label = 1 for real (positive) interactions
label = 0 for sampled negative items

## Preprocesing
Reindex Users and Items for the use of embeddings.

You will train a simple matrix factorization model using learned user and item embeddings.

```
u_vec = embedding_user[user_id]

i_vec = embedding_item[item_id]
```
The predicted score for a user–item pair is the dot product:

`pred(u, i) = u_vec · i_vec`

This score represents how likely the user is to interact with the item.

## Training the Model

You will train the model using the `(user_id, item_id, label)` dataset created above. Use binary cross-entropy loss. Train for several epochs. Go over the full dataset multiple times to allow embeddings to converge. Track the training loss. The loss should generally decrease across epochs. If it doesn’t, adjust: 1) learning rate, 2) number of epochs, 3) embedding dimension.

This code gets the numpy array into tensors.
```
user_tensor = torch.from_numpy(train_user).long()
item_tensor = torch.from_numpy(train_item).long()
label_tensor = torch.from_numpy(train_label).float()
```

## 4. Generating Recommendations
After training:

For each user:
1. Compute predicted score for **all candidate items**.
2. Exclude items already seen in training.
3. Return the **top-K highest scoring items**.

You should implement a function with the signature:
```python
def recommend_model(user_id, k):
    """Return top-k item_ids recommended by the trained model."""
```

Note: In this homework you will compute recommendations by scoring all items and sorting them.
In real production systems (Spotify, YouTube, TikTok, Amazon), this step is performed using Approximate Nearest Neighbor (ANN) search libraries such as FAISS or ScaNN. These libraries find the highest-scoring items in the embedding space without scoring every item, enabling millisecond-level recommendations even with millions of items.

## 5. Evaluation
Use the **same metrics** as in Part 1:

Compare:
- Your trained model
- Popularity baseline
- Item–item similarity baseline

Visualize results in a small table or bar chart.

---

## Deliverables
- Jupyter notebook with:
  - Negative sampling implementation
  - Model training code
  - Recommendation generation
  - Evaluation
  - Comparison with baselines

---

