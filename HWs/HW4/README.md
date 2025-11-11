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

### ✔ Code (Jupyter Notebook)
- Preprocessing steps
- Train/test construction
- Baseline models
- Evaluation functions
- Performance comparison plot or table

---
