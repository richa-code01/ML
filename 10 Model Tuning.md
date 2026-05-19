### 10. Model Tuning (Hyperparameter Tuning)

Model tuning means adjusting a model’s **hyperparameters** to make it perform better on **new/unseen data**.

In simple words:
- We don’t change the dataset
- We don’t change the algorithm
- We change the **hyperparameters** of the algorithm

---

### What Do We Tune?

Hyperparameters are values we choose **before/during training**.

---

### Hyperparameter (Simple Meaning)

A hyperparameter is a value we set before training that controls how the model learns.

Difference:
- **Model parameters** (learned from data) → weights/coefficients/bias
- **Hyperparameters** (chosen by us) → learning settings

Example:
- In KNN, `k` is a hyperparameter
- In Decision Tree, `max_depth` is a hyperparameter

Examples:
- KNN: `k` (number of neighbors)
- Decision Tree: `max_depth`, `min_samples_split`
- SVM: `C`, kernel type, `gamma`
- Regularization: `lambda` in Ridge/Lasso

> Hyperparameters are not learned from data like weights/coefficients.

---

### Goal of Tuning

Main goal:
- improve **generalization** (good performance on validation/test)

Not the goal:
- getting very high **training** accuracy (can cause overfitting)

---

### Basic Tuning Workflow

#### Step 1: Split Data

- Training set → learns the model
- Validation set → selects best hyperparameters
- Test set → final unbiased evaluation

---

#### Step 2: Try Multiple Settings

Example (KNN):

```text
k = 1, 3, 5, 7, 9...
```

Example (SVM):

```text
C = 0.1, 1, 10
kernel = linear, rbf
```

---

#### Step 3: Evaluate on Validation

Choose a metric based on the problem:
- Classification: Accuracy, Precision, Recall, F1-score, AUC
- Regression: MAE, MSE, RMSE, R2

---

#### Step 4: Select Best Hyperparameters

Pick the settings that give the best validation score.

---

#### Step 5: Final Test

After selecting hyperparameters:
- evaluate once on **test set**

> Don’t tune using the test set (data leakage).

---

### Common Tuning Methods

#### 1) Manual Search (Trial-and-Error)

Manual search means:
- try a few values based on intuition/experience
- check validation score
- adjust again (repeat)

Common approach:
1. Start with a reasonable default
2. Change **one hyperparameter at a time**
3. Narrow the range around the best value

Example:

```text
Decision Tree:
max_depth = 2, 4, 6, 8
pick the best validation score
```

Pros:
- simple and fast for small problems

Cons:
- may miss the best combination (not systematic)

#### 2) Cross Validation (CV)

Cross validation is a method to evaluate a model by training it **multiple times** on different parts of the data.

Most common type: **K-Fold Cross Validation**.

---

### K-Fold Cross Validation

Idea:
- split the dataset into `K` equal parts (folds)
- train `K` times
- each time, use a different fold as **validation** and remaining folds as **training**

Important:
- it is the **same model type/class**, but it is **re-trained from scratch** `K` times
- think of it as `K` separate fits (conceptually `K` model instances/objects)

#### Steps

1. Choose `K`

```text
K = 5 (5-fold CV)
```

2. Split data into 5 folds

```text
F1  F2  F3  F4  F5
```

3. Train and validate `K` times

```text
Run 1: Train on F2+F3+F4+F5, Validate on F1
Run 2: Train on F1+F3+F4+F5, Validate on F2
Run 3: Train on F1+F2+F4+F5, Validate on F3
Run 4: Train on F1+F2+F3+F5, Validate on F4
Run 5: Train on F1+F2+F3+F4, Validate on F5
```

4. Take the average validation score

If validation scores are:

```text
0.82, 0.80, 0.84, 0.81, 0.83
```

Final CV score:

$$
CV\ Score=\frac{0.82+0.80+0.84+0.81+0.83}{5}
$$

---

### Why Use Cross Validation?

Pros:
- uses data efficiently (every point becomes validation once)
- more reliable than a single train/validation split

Cons:
- slower (training happens `K` times)

> For classification with imbalanced classes, use **Stratified K-Fold**.

#### 3) Grid Search CV

Grid Search CV means:
- try **all combinations** of hyperparameters from a given grid
- a grid has some values for all the hyperparameters of a given model.
- for each combination, evaluate using **Cross Validation (CV)**
- choose the combination with the best **average CV score**

---

### Steps (Grid Search CV)

1. Create a grid (list of values)

Example (SVM):

```text
C = [0.1, 1, 10]
gamma = [0.01, 0.1]
kernel = [rbf]
```

2. For each combination, do K-Fold CV
- train `K` times (each fold)
- take average score

3. Select best hyperparameters

4. Re-train (refit) using best hyperparameters on full training data

5. Evaluate once on test set

---

### Training Cost

Total model trainings:

$$
Total\ Trainings \approx (\#\ combinations)\times K
$$

So it can be slow when the grid is large.

---

### Pros / Cons

Pros:
- systematic (checks all combinations)
- usually gives a strong baseline

Cons:
- slow for large grids

#### 4) Random Search

Random search means:
- choose a range/list for each hyperparameter
- randomly sample a fixed number of combinations
- evaluate and pick the best one

Example (SVM):

```text
C in [0.001, 0.01, 0.1, 1, 10]
gamma in [0.001, 0.01, 0.1, 1]
```

Instead of trying all pairs, we test only `N` random combinations.

#### Steps

1. Define ranges/distributions for hyperparameters
2. Choose number of trials

```text
N = 20 random combinations
```

3. For each trial:
- train model
- evaluate on validation (or CV)
4. Select best hyperparameters

Pros:
- faster than grid search for large search spaces
- can find good results with fewer trials

Cons:
- not guaranteed to try the best combination

> If you use CV with random search, total trainings become `N × K`.

